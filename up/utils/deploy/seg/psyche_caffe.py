import os
import json
import yaml

try:
    import spring.nart.tools.caffe.count as count
    import spring.nart.tools.caffe.utils.graph as graph
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
except:  # noqa
    print('No module named spring in up/utils/deploy/seg/psyche.py')

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.tocaffe_helper import parse_resize_scale
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY

__all__ = ['PsycheParser_caffe']


@KS_PARSER_REGISTRY.register('psyche_caffe')
class PsycheParser_caffe(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def parse_dataset_param(dataset_cfg):
    def get_transform(transformer_cfg, type_name):
        for cfg in transformer_cfg:
            if cfg['type'] == type_name:
                return cfg
        return None

    dataset_cfg.update(dataset_cfg.get('test', {}))

    kwargs_cfg = up_parser.getdotattr(dataset_cfg, 'dataset.kwargs')
    transformer_cfg = kwargs_cfg['transformer']

    # keep scale consistent with tocaffe input blobs shape
    input_h, input_w = parse_resize_scale(dataset_cfg, 'seg')

    pixel_cfg = get_transform(transformer_cfg, 'normalize')
    pixel_means = up_parser.getdotattr(pixel_cfg, 'kwargs.mean')
    pixel_stds = up_parser.getdotattr(pixel_cfg, 'kwargs.std')
    color_mode = up_parser.getdotattr(kwargs_cfg, 'image_reader.kwargs.color_mode')

    dataset_param = dict()
    dataset_param['input_h'] = input_h
    dataset_param['input_w'] = input_w
    dataset_param['pixel_means'] = [i for i in pixel_means]
    dataset_param['pixel_stds'] = [i for i in pixel_stds]
    dataset_param['is_rgb'] = color_mode == 'RGB'

    return dataset_param


def generate_config(train_cfg):
    if isinstance(train_cfg, str):
        with open(train_cfg) as f:
            train_cfg = yaml.load(f)

    kestrel_param = dict()
    kestrel_param['type'] = 'segmentation'
    # dataset param
    assert 'dataset' in train_cfg, 'config file incomplete: lack dataset'
    dataset_param = parse_dataset_param(train_cfg['dataset'])
    kestrel_param.update(dataset_param)
    return kestrel_param


def process_net(prototxt, model, input_h, input_w, input_channel=3):
    # get net
    net, withBinFile = graph.readNetStructure(prototxt, model)
    # update input dim
    scaffold.update_input_dim(net, 0, [1, input_channel, input_h, input_w])
    # merge bn
    net = scaffold.merge_bn(net)

    # get net info
    net_info = dict()
    net_graph = graph.gen_graph(net)
    # get out info
    out_info = dict()

    assert len(net_graph.root) == 1
    net_info['data'] = net_graph.root[0].content.bottom[0]

    # get output blob shape
    _, _, blob_shape = count.inferNet(net)

    # select mask output
    mask = list()
    out_info['output'] = list()
    for leaf in net_graph.leaf:
        top_key = leaf.content.top[0]
        out_item = dict()
        out_item['name'] = top_key
        _, c, h, w = blob_shape[top_key]
        out_item['height'] = int(h)
        out_item['width'] = int(w)
        out_item['channel'] = int(c)
        mask.append(top_key)
        out_info['output'].append(out_item)

    net_info['output'] = list()
    for i in range(len(mask)):
        net_info['output'].append(mask[i])

    return net, net_info, out_info


def generate_common_param(net_info, max_batch_size):
    common_param = dict()
    net_param = dict()
    net_param['net'] = net_info['packname']
    net_param['backend'] = net_info['backend']
    net_param['max_batch_size'] = max_batch_size
    net_param['input'] = {'data': net_info['data']}

    net_param['output'] = dict()
    for i, pair in enumerate(net_info['output']):
        net_param['output']['blob_pred'] = pair
    common_param['net'] = net_param
    return common_param


@KS_PROCESSOR_REGISTRY.register('psyche_caffe')
class PsycheProcessor_caffe(BaseProcessor):
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

        net, net_info, out_info = process_net(
            self.prototxt, self.model, kestrel_param['input_h'], kestrel_param['input_w'], self.input_channel)
        net_info['packname'] = 'model'
        net_info['backend'] = 'kestrel_caffe'
        kestrel_param.update(out_info)
        # save model
        scaffold.generate_model(net, self.save_path, net_info['packname'])

        common_param = generate_common_param(net_info, self.max_batch_size)
        kestrel_param['model_files'] = common_param

        scaffold.generate_json_file(os.path.join(self.save_path, 'parameters.json'), kestrel_param)
        scaffold.generate_meta(self.save_path, self.name, 'psyche', version)
        pack_list = [net_info['packname']]
        scaffold.compress_model(self.save_path, pack_list, self.name, version)
