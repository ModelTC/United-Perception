import os
import json
import shutil
from onnx import TensorProto

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
from up.utils.deploy.det.sphinx import generate_config


@KS_PARSER_REGISTRY.register('sphinx_onnx')
class SphinxParser_onnx(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def split_net(model, input_h, input_w, input_channel=3):
    G = onnx.load(model)
    G = auto_remove_cast(G)
    G_nart = OnnxDecoder().decode(G)

    wmap = build_name_dict(G.graph.initializer)
    vimap = build_name_dict(G.graph.value_info)

    graph_output_names = [out.name for out in G.graph.output]
    roi_node_lists = []
    for node in G.graph.node:
        if 'Roi' in node.name:
            roi_node_lists.append(node)

    # rpn
    rpn_inputs_names = [inp.name for inp in G.graph.input]
    rpn_outputs_names = [name for name in graph_output_names if 'RPN' in name]

    rpn_nodes = collect_subnet_nodes(G, rpn_inputs_names, rpn_outputs_names)
    rpn_inits, rpn_values = collect_subnet_inits_values(rpn_nodes, wmap, vimap)

    # make rpn inputs & outputs
    rpn_data = onnx.helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, input_channel, input_h, input_w])
    rpn_inputs = [rpn_data]
    rpn_outputs = [output for output in G.graph.output if 'RPN' in output.name]

    rpn_graph = onnx.helper.make_graph(rpn_nodes, name='rpn', inputs=rpn_inputs, outputs=rpn_outputs,
                                       initializer=rpn_inits, value_info=rpn_values)
    rpn_model = onnx.helper.make_model(rpn_graph)
    rpn_model_path = model.replace(model.split('/')[-1], 'rpn.onnx')
    onnx.save(rpn_model, rpn_model_path)
    logger.info("save rpn to {}".format(rpn_model_path))

    # det
    det_input_names_list = []
    for node in roi_node_lists:
        det_input_names_list += node.input

    det_output_names = [name for name in graph_output_names if 'Bbox' in name]
    det_nodes = collect_subnet_nodes(G, det_input_names_list, det_output_names)
    det_inits, det_values = collect_subnet_inits_values(det_nodes, wmap, vimap)
    # make det inputs & outputs
    det_inputs = []
    for i in range(len(roi_node_lists)):
        feat_shape = G_nart.get_tensor_shape(roi_node_lists[i].input[0])
        det_feat = onnx.helper.make_tensor_value_info(
            roi_node_lists[i].input[0], TensorProto.FLOAT, feat_shape)
        det_roi = onnx.helper.make_tensor_value_info(roi_node_lists[i].input[1], TensorProto.FLOAT, [1, 5, 1, 1])
        det_inputs.append(det_feat)
        det_inputs.append(det_roi)

    det_inputs_names = [inp.name for inp in det_inputs]
    det_outputs = [output for output in G.graph.output if 'Bbox' in output.name]
    det_outputs_names = [out.name for out in det_outputs]
    det_graph = onnx.helper.make_graph(det_nodes, name='det', inputs=det_inputs, outputs=det_outputs,
                                       initializer=det_inits, value_info=det_values)
    det_model = onnx.helper.make_model(det_graph)
    det_model_path = model.replace(model.split('/')[-1], 'det.onnx')
    logger.info("save det to {}".format(det_model_path))
    onnx.save(det_model, det_model_path)

    # rpn net info
    rpn_net_info = {}
    rpn_net_info['net'] = "rpn.onnx"
    rpn_net_info['backend'] = 'kestrel_nart'
    # rpn input
    rpn_net_info['input'] = {'data': rpn_inputs_names[0]}
    # rpn output
    rpn_net_info['output'] = {}
    for i in range(len(rpn_outputs_names) // 2):
        rpn_net_info['output'].update({"bbox_{}".format(i): rpn_outputs_names[i + 5]})
        rpn_net_info['output'].update({"score_{}".format(i): rpn_outputs_names[i]})

    # det net info
    det_net_info = {}
    det_net_info['net'] = "det.onnx"
    det_net_info['backend'] = 'kestrel_nart'
    # det input & rpn marked output
    det_net_info['input'] = {}
    rpn_net_info['marked_output'] = {}
    for i in range(len(det_inputs_names) // 2):
        det_net_info['input'].update({"feature_{}".format(i): det_inputs_names[i * 2]})
        rpn_net_info['marked_output'].update({"shared_rpn_layer_{}".format(i): det_inputs_names[i * 2]})
        det_net_info['input'].update({"roi_{}".format(i): det_inputs_names[i * 2 + 1]})
    # det output
    det_net_info['output'] = {"score": det_outputs_names[0], "bbox": det_outputs_names[1]}
    onnx_net_info = {'rpn_net': rpn_net_info, 'det_net': det_net_info}

    return rpn_model_path, det_model_path, onnx_net_info


def process_net(model, input_h, input_w, input_channel):
    # merge bn
    model = merged_bn(model)

    # process reshape
    model = convert_onnx_for_mbs(model)

    # get net & split net
    rpn_net, det_net, net_info = split_net(model, input_h, input_w, input_channel)

    return rpn_net, det_net, net_info


def generate_common_param(net_info, kestrel_param, max_batch_size):
    net_info['rpn_net']['max_batch_size'] = max_batch_size
    net_info['det_net']['max_batch_size'] = max_batch_size * kestrel_param['aft_top_k']
    return net_info


@KS_PROCESSOR_REGISTRY.register('sphinx_onnx')
class SphinxProcessor_onnx(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        # get resize hw if provided
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['short_scale'] = h
            kestrel_param['long_scale'] = w

        model = 'tocaffe/model.onnx'
        rpn_net, det_net, net_info = process_net(model,
                                                 kestrel_param['short_scale'],
                                                 kestrel_param['long_scale'],
                                                 self.input_channel)

        # move onnx
        if os.path.exists(self.save_path):
            os.system('rm -rf {}'.format(self.save_path))
        os.mkdir(self.save_path)
        shutil.move(rpn_net, self.save_path)
        shutil.move(det_net, self.save_path)

        if self.serialize:
            assert NotImplementedError

        common_param = generate_common_param(net_info, kestrel_param, self.max_batch_size)
        kestrel_param['model_files'] = common_param

        scaffold.generate_json_file(os.path.join(self.save_path, 'parameters.json'), kestrel_param)

        scaffold.generate_meta(self.save_path, self.name, 'sphinx', version,
                               {'class': kestrel_param['class']})

        scaffold.compress_model(self.save_path, ['rpn.onnx', 'det.onnx'],
                                self.save_path, version)
