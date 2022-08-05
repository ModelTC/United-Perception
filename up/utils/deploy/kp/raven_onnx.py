import os
import yaml
import onnx

from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from up.utils.deploy.kp.raven import generate_parameter, generate_config
from up.utils.deploy import parser as up_parser

import spring.nart.tools.kestrel.utils.scaffold as scaffold
from spring.nart.utils.onnx_utils.onnx_builder import OnnxDecoder
from spring.nart.core.graph import Node, Model

__all__ = ['RavenParser_onnx', 'RavenProcessor_onnx']


@KS_PARSER_REGISTRY.register('raven_onnx')
class RavenParser_onnx(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def process_onnx_net(onnx_path, save_path, input_h, input_w, num_kpts=17):

    with open(onnx_path, "rb") as f:
        onf = onnx.load(f)
    x = OnnxDecoder()
    graph = x.decode(onf)  # <class 'spring.nart.core.graph.Graph'>
    graph.del_output('blob_pred')  # del output by name

    last_node = graph.nodes[-1]
    last_idx = len(graph.nodes)

    # branch1
    slice_fg = Node.make_node(
        'slice_fg',
        op_type='Split',
        input=last_node.output,
        output=[
            'slice_fg_out',
            'slice_bg_out'])

    axis = onnx.helper.make_attribute(key='axis', value=1)
    slice_fg.add_attribute(axis)

    split = onnx.helper.make_attribute(key='split', value=[num_kpts, 1])
    slice_fg.add_attribute(split)

    sigmoid_fg = Node.make_node('sigmoid_fg', op_type='Sigmoid', input=['slice_fg_out'], output=['sigmoid_fg_out'])

    heatmap2coord_fg = Node.make_node(
        'heatmap2coord',
        op_type='HeatMap2Coord',
        input=sigmoid_fg.output,
        output=['heatmap2coord_out'])
    coord_h = onnx.helper.make_attribute(key='coord_h', value=input_h)
    coord_w = onnx.helper.make_attribute(key='coord_w', value=input_w)
    coord_reposition = onnx.helper.make_attribute(key='coord_reposition', value=True)
    heatmap2coord_fg.add_attribute(coord_h)
    heatmap2coord_fg.add_attribute(coord_w)
    heatmap2coord_fg.add_attribute(coord_reposition)

    graph.insert_nodes([slice_fg, sigmoid_fg, heatmap2coord_fg], last_idx)

    # branch2
    sigmoid_vis = Node.make_node('sigmoid_vis', op_type='Sigmoid', input=last_node.output, output=['sigmoid_vis_out'])
    graph.insert_nodes([sigmoid_vis], last_idx)

    graph.add_output('sigmoid_vis_out')
    graph.add_output('heatmap2coord_out')

    model = Model.make_model(graph)
    model = model.dump_to_onnx()
    model_file = os.path.join(save_path, 'model.onnx')

    onnx.save(model, model_file)

    net_info = {
        "net": "model.onnx",
        "backend": "kestrel_nart",
        "data": "data",
        "point": "heatmap2coord_out",
        "score": "sigmoid_vis_out"
    }
    return net_info


def generate(model, path, name, max_batch_size, cfg_params, version, serialize=False):
    packname = 'model.onnx'

    if os.path.exists("./{}".format(path)):
        os.system('rm -rf {}'.format("./{}".format(path)))
    os.mkdir(path)

    net_info = process_onnx_net(model, path, cfg_params['structure']['input_h'], cfg_params['structure']['input_w'],
                                cfg_params['structure']['num_kpts'])

    # serialize model
    if serialize:
        assert NotImplementedError

    # generate meta
    scaffold.generate_meta(path, name, 'raven', version)

    # generate parameter
    generate_parameter(path, packname, max_batch_size, net_info, cfg_params)

    # compress and save
    scaffold.compress_model(path, [packname], name, version)


@KS_PROCESSOR_REGISTRY.register('raven_onnx')
class RavenProcessor_onnx(BaseProcessor):
    def process(self):
        version = scaffold.check_version_format(self.version)

        # get param from up config
        with open(self.kestrel_param_json) as f:
            cfg_params = yaml.load(f)
        assert cfg_params['common']['bg']

        model = 'tocaffe/model.onnx'
        generate(model, self.save_path, self.name, self.max_batch_size, cfg_params, version, self.serialize)
