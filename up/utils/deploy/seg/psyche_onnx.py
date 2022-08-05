import os
import json
import shutil
import onnx

try:
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
    from spring.nart.utils.onnx_utils.onnx_builder import OnnxDecoder
except:  # noqa
    print('No module named spring in up/utils/deploy/seg/psyche.py')

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from up.utils.general.latency_helper import merged_bn, convert_onnx_for_mbs
from up.utils.deploy.seg.psyche import generate_common_param, generate_config

__all__ = ['PsycheParser_onnx', 'PsycheProcessor_onnx']


@KS_PARSER_REGISTRY.register('psyche_onnx')
class PsycheParser_onnx(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def process_net(model):
    # merge bn
    model = merged_bn(model)

    # process reshape
    model = convert_onnx_for_mbs(model)

    # get net info
    G = onnx.load(model)
    G_nart = OnnxDecoder().decode(G)

    net_info = dict()
    net_info['data'] = G.graph.input[0].name

    out_info = dict()
    # select mask output
    mask = list()
    out_info['output'] = list()
    for oup in G.graph.output:
        out_item = dict()
        out_item['name'] = oup.name
        _, c, h, w = G_nart.get_tensor_shape(oup.name)
        out_item['height'] = int(h)
        out_item['width'] = int(w)
        out_item['channel'] = int(c)
        mask.append(oup.name)
        out_info['output'].append(out_item)

    net_info['output'] = list()
    for i in range(len(mask)):
        net_info['output'].append(mask[i])

    return model, net_info, out_info


@KS_PROCESSOR_REGISTRY.register('psyche_onnx')
class PsycheProcessor_onnx(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        if self.input_channel != 3:
            kestrel_param['rgb_flag'] = False
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['input_h'] = h
            kestrel_param['input_w'] = w

        model = 'tocaffe/model.onnx'
        net, net_info, out_info = process_net(model)
        net_info['packname'] = 'model.onnx'
        net_info['backend'] = 'kestrel_nart'
        kestrel_param.update(out_info)

        # move onnx model
        if os.path.exists(self.save_path):
            os.system('rm -rf {}'.format(self.save_path))
        os.mkdir(self.save_path)
        shutil.move("tocaffe/model.onnx", self.save_path)

        common_param = generate_common_param(net_info, self.max_batch_size)
        kestrel_param['model_files'] = common_param

        scaffold.generate_json_file(os.path.join(self.save_path, 'parameters.json'), kestrel_param)
        scaffold.generate_meta(self.save_path, self.name, 'psyche', version)
        pack_list = [net_info['packname']]
        scaffold.compress_model(self.save_path, pack_list, self.save_path, version)
