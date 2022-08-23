import os
import json
import shutil
import onnx

try:
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
except Exception as err:
    print(err)

from up.utils.deploy import parser as up_parser
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.latency_helper import merged_bn, convert_onnx_for_mbs
from up.utils.deploy.det.essos_caffe import generate_common_param, generate_config


__all__ = ['EssosParser', 'EssosProcessor']


@KS_PARSER_REGISTRY.register('essos')
class EssosParser(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def process_net(model):
    # merge bn
    model_merged = merged_bn(model)

    # process reshape
    model_fixed = convert_onnx_for_mbs(model_merged)

    # get net info
    net_info = dict()
    G = onnx.load(model_fixed)

    net_info['data'] = G.graph.input[0].name

    # select score and bbox output
    score = list()
    bbox = list()
    for node in G.graph.node:
        for oup in node.output:
            if 'cls' in oup:
                score.append(oup)
            elif 'loc' in oup:
                bbox.append(oup)
            else:
                continue

    score.sort(key=lambda x: int(x[-1]), reverse=False)
    bbox.sort(key=lambda x: int(x[-1]), reverse=False)

    net_info['output'] = list()
    for i in range(len(bbox)):
        net_info['output'].append([score[i], bbox[i]])

    return model_fixed, net_info


@KS_PROCESSOR_REGISTRY.register('essos')
class EssosProcessor(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        if self.input_channel != 3:
            kestrel_param['rgb_flag'] = False

        # process_net
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['short_scale'] = h
            kestrel_param['long_scale'] = w

        model = 'tocaffe/model.onnx'
        model_fixed, net_info = process_net(model)

        net_info['packname'] = model_fixed.split('/')[-1]
        net_info['backend'] = 'kestrel_nart'

        if self.serialize:
            assert NotImplementedError
        if self.nnie:
            assert NotImplementedError

        common_param, rpn_blob_param = generate_common_param(net_info, self.max_batch_size)
        kestrel_param['model_files'] = common_param
        kestrel_param['rpn_blobs'] = rpn_blob_param

        # move onnx
        if os.path.exists(self.save_path):
            os.system('rm -rf {}'.format(self.save_path))
        os.mkdir(self.save_path)
        shutil.move(model_fixed, self.save_path)

        scaffold.generate_json_file(os.path.join(self.save_path, 'parameters.json'), kestrel_param)
        scaffold.generate_meta(self.save_path, self.name, 'essos', version, {'class': kestrel_param['class']})
        pack_list = [net_info['packname']]
        if self.nnie:
            assert NotImplementedError
        scaffold.compress_model(self.save_path, pack_list, self.name, version)
