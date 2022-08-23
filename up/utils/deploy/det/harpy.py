import os
import json
from onnx import TensorProto
import shutil

try:
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
    from spring.nart.utils.onnx_utils.onnx_builder import OnnxDecoder
except Exception as err:
    print(err)

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from up.utils.general.latency_helper import merged_bn, convert_onnx_for_mbs
from up.utils.deploy.onnx_utils import *
from up.utils.deploy.det.harpy_caffe import generate_common_param, generate_config


@KS_PARSER_REGISTRY.register('harpy')
class HarpyParser(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def split_net(
        model,
        anchor_num,
        cls_channel_num,
        input_h,
        input_w,
        input_channel=3):
    # load model and remove useless cast layers
    G = onnx.load(model)
    G = auto_remove_cast(G)
    G_nart = OnnxDecoder().decode(G)

    wmap = build_name_dict(G.graph.initializer)
    vimap = build_name_dict(G.graph.value_info)

    graph_output_names = [out.name for out in G.graph.output]
    for node in G.graph.node:
        if 'Roi' in node.name:
            roi_node = node
            break

    # rpn
    rpn_inputs_names = [inp.name for inp in G.graph.input]
    rpn_outputs_names = [name for name in graph_output_names if 'RPN' in name]

    rpn_nodes = collect_subnet_nodes(G, rpn_inputs_names, rpn_outputs_names)
    rpn_inits, rpn_values = collect_subnet_inits_values(rpn_nodes, wmap, vimap)
    # make rpn inputs & outputs
    rpn_data = onnx.helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, input_channel, input_h, input_w])
    rpn_inputs = [rpn_data]
    rpn_outputs = [output for output in G.graph.output if 'RPN' in output.name]

    # find last reshape
    for idx, node in enumerate(rpn_nodes):
        if node.op_type == 'Reshape' and 'RPNPostProcess' in node.output[0]:
            last_reshape_node = node
            last_idx = idx
            break
    rpn_nodes.remove(last_reshape_node)

    last_const_idx = len(rpn_nodes)
    last_const_node = rpn_nodes[-1]

    # fix cls output
    new_const_node = make_constant_dims(last_reshape_node.input[1], [1, anchor_num, cls_channel_num, -1])
    rpn_nodes.remove(last_const_node)
    rpn_nodes.insert(last_const_idx, new_const_node)

    new_reshape = onnx.helper.make_node(name=last_reshape_node.name,
                                        op_type=last_reshape_node.op_type,
                                        inputs=last_reshape_node.input,
                                        outputs=['last_transpose_inp'])
    last_transpose = onnx.helper.make_node(name='last_transpose',
                                           op_type='Transpose',
                                           inputs=['last_transpose_inp'],
                                           outputs=last_reshape_node.output)
    perm = onnx.helper.make_attribute(key='perm', value=[0, 2, 1, 3])
    last_transpose.attribute.append(perm)
    rpn_nodes.insert(last_idx, new_reshape)
    rpn_nodes.insert(last_idx + 1, last_transpose)

    # make rpn graph
    rpn_graph = onnx.helper.make_graph(rpn_nodes, name='rpn', inputs=rpn_inputs, outputs=rpn_outputs,
                                       initializer=rpn_inits, value_info=rpn_values)
    rpn_model = onnx.helper.make_model(rpn_graph)
    rpn_model_path = model.replace(model.split('/')[-1], 'rpn.onnx')
    onnx.save(rpn_model, rpn_model_path)

    # det
    det_input_names = roi_node.input
    det_output_names = [name for name in graph_output_names if 'Bbox' in name]
    det_nodes = collect_subnet_nodes(G, det_input_names, det_output_names)
    det_inits, det_values = collect_subnet_inits_values(det_nodes, wmap, vimap)

    # make det inputs & outputs
    feat_shape = G_nart.get_tensor_shape(roi_node.input[0])
    det_feats = onnx.helper.make_tensor_value_info(roi_node.input[0], TensorProto.FLOAT, feat_shape)
    det_rois = onnx.helper.make_tensor_value_info(roi_node.input[1], TensorProto.FLOAT, [1, 5, 1, 1])

    det_inputs = [det_feats, det_rois]
    det_outputs = [output for output in G.graph.output if 'Bbox' in output.name]
    det_outputs_names = [out.name for out in det_outputs]
    det_graph = onnx.helper.make_graph(det_nodes, name='rpn', inputs=det_inputs, outputs=det_outputs,
                                       initializer=det_inits, value_info=det_values)
    det_model = onnx.helper.make_model(det_graph)
    det_model_path = model.replace(model.split('/')[-1], 'det.onnx')
    onnx.save(det_model, det_model_path)

    onnx_net_info = {
        'rpn': {
            'data': rpn_inputs_names[0],
            'score': rpn_outputs_names[0],
            'bbox': rpn_outputs_names[1],
            'feature': roi_node.input[0],
        },
        'det': {
            'feature': roi_node.input[0],
            'roi': roi_node.input[1],
            'score': det_outputs_names[0],
            'bbox': det_outputs_names[1],
        }
    }
    return rpn_model_path, det_model_path, onnx_net_info


def process_net(model, anchor_num, rpn_cls_channel_num, input_h, input_w, input_channel):
    # merge bn
    model = merged_bn(model)

    # process reshape
    model = convert_onnx_for_mbs(model)

    # split rpn & det and get input/output info
    rpn_net, det_net, net_info = split_net(model, anchor_num, rpn_cls_channel_num, input_h, input_w, input_channel)

    return rpn_net, det_net, net_info


@KS_PROCESSOR_REGISTRY.register('harpy')
class HarpyProcessor(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)

        # get param from pod config
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        # get resize hw if provided
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['short_scale'] = h
            kestrel_param['long_scale'] = w
        # process_net
        # number of channel for classification
        if kestrel_param.get('sigmoid', False):
            offset = 0
        else:
            offset = 1
        rpn_cls_channel_num = 1 + offset

        model = 'tocaffe/model.onnx'
        rpn_net, det_net, net_info = process_net(model,
                                                 kestrel_param['num_anchors'],
                                                 rpn_cls_channel_num,
                                                 kestrel_param['short_scale'],
                                                 kestrel_param['long_scale'],
                                                 self.input_channel)
        net_info['rpn']['packname'] = 'rpn.onnx'
        net_info['det']['packname'] = 'det.onnx'
        net_info['rpn']['backend'] = 'kestrel_nart'
        net_info['det']['backend'] = 'kestrel_nart'

        # move onnx
        if os.path.exists(self.save_path):
            os.system('rm -rf {}'.format(self.save_path))
        os.mkdir(self.save_path)
        shutil.move(rpn_net, self.save_path)
        shutil.move(det_net, self.save_path)

        common_param = generate_common_param(net_info, self.max_batch_size,
                                             kestrel_param['aft_top_k'])

        kestrel_param['model_files'] = common_param

        if self.serialize:
            assert NotImplementedError

        scaffold.generate_json_file(
            os.path.join(self.save_path, 'parameters.json'), kestrel_param)

        scaffold.generate_meta(
            self.save_path, self.name, 'harpy', version, {'class': kestrel_param['class']})

        scaffold.compress_model(
            self.save_path, ['rpn.onnx', 'det.onnx'], self.name, version)
