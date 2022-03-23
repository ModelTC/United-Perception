from __future__ import division

# Standard Library
import json


# Import from up
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import LATENCY_REGISTRY


def merged_bn(path_model):
    import onnx
    from spring.nart.utils.onnx_utils import OnnxDecoder
    from spring.nart.passes import DeadCodeElimination, ConvFuser, GemmFuser
    from spring.nart.core import Model
    model = onnx.load(path_model)
    graph = OnnxDecoder().decode(model)
    graph.update_tensor_shape()
    graph.update_topology()

    ConvFuser().run(graph)
    GemmFuser().run(graph)
    DeadCodeElimination().run(graph)
    graph.update_topology()

    model = Model.make_model(graph)
    model = model.dump_to_onnx()

    model_file = path_model.replace('.onnx', '_merged.onnx')
    onnx.save(model, model_file)
    return model_file


def convert_onnx_for_mbs(onnx_file):
    import onnx
    from onnx import numpy_helper

    def get_shape(node):
        attrs = {attr.name: attr for attr in node.attribute}
        shapes = numpy_helper.to_array(attrs['value'].t).reshape(-1).tolist()
        return shapes

    def make_constant_dims(name, shapes):
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value=onnx.helper.make_tensor("val", onnx.TensorProto.INT64, [len(shapes), ], shapes),
        )
        return node

    G = onnx.load(onnx_file)
    consts = {}
    for idx, node in enumerate(G.graph.node):
        node = G.graph.node[idx]
        if node.op_type == 'Constant':
            consts[node.output[0]] = (idx, node)
        if node.op_type == 'Reshape':
            cid, cnode = consts[node.input[1]]
            ori_shape = get_shape(cnode)
            if len(ori_shape) == 2 and -1 in ori_shape[1:]:
                logger.info('Replace Reshape [{}] to Flatten'.format(node.name))
                flatten_node = onnx.helper.make_node("Flatten", inputs=[node.input[0]], outputs=node.output, axis=1)
                G.graph.node.remove(node)
                G.graph.node.insert(idx, flatten_node)
            else:
                new_shape = [-1] + ori_shape[1:]
                logger.info('Reshape [{}] shape from {} to {}'.format(node.name, ori_shape, new_shape))
                new_cnode = make_constant_dims(node.input[1], new_shape)
                G.graph.node.remove(cnode)
                G.graph.node.insert(cid, new_cnode)

    onnx_save = onnx_file.split('.')[0] + "_fix.onnx"
    onnx.save(G, onnx_save)
    logger.info("Saved onnx to {}".format(onnx_save))
    return onnx_save


@LATENCY_REGISTRY.register('base')
class BaseLatency(object):
    def __init__(self, cfg_gdbp, onnx_file):
        self.cfg_gdbp = cfg_gdbp
        onnx_file_fix = convert_onnx_for_mbs(onnx_file)
        logger.info(f'gdbp:{onnx_file_fix}')
        self.onnx_file = onnx_file_fix

    def process(self):
        from spring.models.latency import Latency

        hardware_name = self.cfg_gdbp.get('hardware_name', 'cpu')
        backend_name = self.cfg_gdbp.get('backend_name', 'ppl2')
        data_type = self.cfg_gdbp.get('data_type', 'fp32')
        batch_size = self.cfg_gdbp.get('batch_size', 1)
        res_json = self.cfg_gdbp.get('res_json', 'latency_test.json')

        # merge bn
        logger.info(f'merge bn')
        logger.info(self.onnx_file)
        model_file = merged_bn(self.onnx_file)
        # latency test
        logger.info(f'start latency test')
        latency_client = Latency()
        ret_val = latency_client.call(
            hardware_name,
            backend_name,
            data_type,
            batch_size,
            model_file,
            graph_name="imagenet_resnet",
            force_test=False,
            print_info=False,
            match_mode=1,
            match_speed_mode=1,
            match_speed_async=True
        )
        logger.info(f'ret_val:{ret_val}')
        # save latency test results
        res_json = open(res_json, 'w')
        res_json.write(json.dumps(ret_val))
        res_json.close()
