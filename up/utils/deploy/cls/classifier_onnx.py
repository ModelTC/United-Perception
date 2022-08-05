import json
import os
import shutil
import onnx

try:
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
except Exception as err:
    print(err)

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from up.utils.general.latency_helper import merged_bn, convert_onnx_for_mbs
from up.utils.deploy.cls.classifier import generate_parameter, generate_config

__all__ = ['ClassifierParser_onnx', 'ClassifierProcessor_onnx']


@KS_PARSER_REGISTRY.register('classifier_onnx')
class ClassifierParser_onnx(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def generate(model, path, name, serialize, max_batch_size, cfg_params, version):
    # merge bn
    model = merged_bn(model)
    # convert for mbs
    model = convert_onnx_for_mbs(model)
    packname = model.split('/')[-1]

    # get net info
    net_info = dict()
    G = onnx.load(model)
    net_info['data'] = G.graph.input[0].name
    net_info['score'] = G.graph.output[0].name
    net_info['net'] = packname
    # serialize model
    if serialize:
        assert NotImplementedError
    else:
        net_info['backend'] = 'kestrel_nart'
    net_info['max_batch_size'] = max_batch_size

    # get data info
    inp = G.graph.input[0]
    net_input_dim = [dim_value.dim_value for dim_value in inp.type.tensor_type.shape.dim]
    net_info['input_h'] = net_input_dim[2]
    net_info['input_w'] = net_input_dim[3]

    # move onnx
    if os.path.exists(path):
        os.system('rm -rf {}'.format(path))
    os.mkdir(path)
    shutil.move(model, path)

    # generate meta
    scaffold.generate_meta(path, name, 'classifier', version)
    generate_parameter(path, packname, max_batch_size, net_info, cfg_params)

    # compress and save
    scaffold.compress_model(path, [packname, 'category_param.json'], name, version)


@KS_PROCESSOR_REGISTRY.register('classifier_onnx')
class ClassifierProcessor_onnx(BaseProcessor):
    def process(self):
        version = scaffold.check_version_format(self.version)

        # get param from up config
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        model = 'tocaffe/model.onnx'
        generate(model, self.save_path, self.name, False, 8, kestrel_param, version)
