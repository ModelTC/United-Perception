import os
import copy
import yaml

from up.utils.general.user_analysis_helper import get_task_from_cfg
from up.utils.general.tocaffe_helper import parse_resize_scale
from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY

import spring.nart.tools.kestrel.utils.scaffold as scaffold
import spring.nart.tools.kestrel.utils.net_transform as transform
import spring.nart.tools.caffe.utils.graph as graph
import spring.nart.tools.caffe.count as count

__all__ = ['RavenParser', 'RavenProcessor']


@KS_PARSER_REGISTRY.register('raven')
class RavenParser(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def generate_config(train_cfg):
    if isinstance(train_cfg, str):
        with open(train_cfg) as f:
            train_cfg = yaml.load(f)

    kestrel_param = dict()
    kestrel_param['common'] = dict()
    # update bg
    net_info = train_cfg.get('net', [])
    kestrel_param['common']['bg'] = False
    for net in net_info:
        if 'has_bg' in net.get('kwargs', {}):
            kestrel_param['common']['bg'] = net['kwargs']['has_bg']
    assert kestrel_param['common']['bg'], 'common.bg should be True.'
    # update structure
    kestrel_param['structure'] = dict()
    if train_cfg['to_kestrel'].get('resize_hw', '') != '':
        resize_hw = train_cfg['to_kestrel'].get('resize_hw', '640x1024')
        input_h, input_w = (int(i) for i in resize_hw.strip().split("x"))
    else:
        task_type = get_task_from_cfg(train_cfg)
        input_h, input_w = parse_resize_scale(train_cfg['dataset'], task_type)
    kestrel_param['structure']['input_h'] = input_h
    kestrel_param['structure']['input_w'] = input_w
    kestrel_param['structure']['num_kpts'] = train_cfg.get('num_kpts', 17)

    # update test
    kestrel_param['test'] = dict()
    dataset_cfg = copy.deepcopy(train_cfg['dataset'])
    dataset_cfg.update(dataset_cfg.get('test', {}))
    kestrel_param['test']['box_scale'] = dataset_cfg['dataset']['kwargs'].get('box_scale', 1.0)
    # update crop_square, keep_aspect_ratio
    # img_norm_factor, means, std
    kestrel_param['common']['crop_square'] = False
    kestrel_param['common']['keep_aspect_ratio'] = False
    kestrel_param['common']['img_norm_factor'] = 255.0
    kestrel_param['common']['means'] = [0.485, 0.456, 0.406]
    kestrel_param['common']['stds'] = [0.229, 0.224, 0.225]
    transforms = dataset_cfg['dataset']['kwargs']['transformer']
    for tf in transforms:
        if 'crop_square' in tf['type']:
            kestrel_param['common']['crop_square'] = True
        if 'keep_aspect_ratio' in tf['type']:
            kestrel_param['common']['keep_aspect_ratio'] = True
        if 'means' in tf.get('kwargs', {}):
            kestrel_param['common']['means'] = tf['kwargs']['means']
        if 'stds' in tf.get('kwargs', {}):
            kestrel_param['common']['stds'] = tf['kwargs']['stds']
        if 'img_norm_factor' in tf.get('kwargs', {}):
            kestrel_param['common']['img_norm_factor'] = tf['kwargs']['img_norm_factor']
    return kestrel_param


def get_spatial(net, node):
    tmp_node = node
    _, _, infer_shape = count.inferNet(net)
    while True:
        if tmp_node.content.type == 'Pooling':
            spatial = infer_shape[tmp_node.prev[0].content.top[0]][2]
            break
        tmp_node = tmp_node.prev[0]
    return spatial


def remove_branch(net, node):
    tmp = node
    while True:
        print('remove useless layer: {}'.format(tmp.content.name))
        net.layer.remove(tmp.content)
        if len(tmp.prev[0].succ) > 1:
            break
        tmp = tmp.prev[0]
    return net


def add_layer(net, net_graph, node, input_h, input_w, num_kpts):
    _ = transform.add_sigmoid(net, net_graph, node.content, 'visibility', insert=False)
    tmp = node
    while tmp.content.type != 'Convolution':
        tmp = tmp.prev[0]
    tmp = transform.add_slice(net, net_graph, tmp.content, 'fg', slice_point=num_kpts, axis=1, insert=False)
    tmp = transform.add_sigmoid(net, net_graph, tmp, 'fg', insert=False)
    _ = transform.add_heatmap2coord(net, net_graph, tmp, 'fg', input_h, input_w, coord_reposition=True, insert=False)

    return net


def generate_parameter(path, packname, max_batch_size, net_info, cfg_params):
    param = dict()
    param['model_files'] = dict()
    net = dict()
    net['net'] = packname
    net['backend'] = net_info['backend']
    net['max_batch_size'] = max_batch_size
    net['input'] = {'data': net_info['data']}
    net['output'] = {'point': net_info['point'], 'score': net_info['score']}
    if net_info.get('background', None):
        net['output']['background'] = net_info['background']
    param['model_files']['net'] = net
    param['input_h'] = cfg_params['structure']['input_h']
    param['input_w'] = cfg_params['structure']['input_w']
    param['padding'] = [(cfg_params['test']['box_scale'] - 1) / 2] * 4
    param['square'] = cfg_params['common']['crop_square']
    param['keep_aspect_ratio'] = cfg_params['common']['keep_aspect_ratio']
    param['pixel_means'] = [i * cfg_params['common']['img_norm_factor'] for i in cfg_params['common']['means']]
    param['pixel_stds'] = [i * cfg_params['common']['img_norm_factor'] for i in cfg_params['common']['stds']]
    param['is_rgb'] = True
    scaffold.generate_json_file(os.path.join(path, 'parameters.json'), param)


def process_net(net, input_h, input_w, num_kpts):
    net_graph = graph.gen_graph(net)

    # find branch with max_spatial
    idx = -1
    max_spatial = -1
    for i, node in enumerate(net_graph.leaf):
        spatial = get_spatial(net, node)
        if spatial > max_spatial:
            idx = i
            max_spatial = spatial

    # update net structure
    for i, node in enumerate(net_graph.leaf):
        if i == idx:
            net = add_layer(net, net_graph, node, input_h, input_w, num_kpts)
        else:
            net = remove_branch(net, node)


def generate(net, path, name, serialize, max_batch_size, cfg_params, version):
    packname = 'model'
    # generate model and save to path
    model_path = scaffold.generate_model(net, path, packname)

    net_info = dict()
    # serialize model
    if serialize:
        packname = 'engine.bin'
        engine_path = os.path.join(path, packname)
        scaffold.serialize(model_path, max_batch_size, engine_path)
        net_info['backend'] = 'kestrel_mixnet'
    else:
        net_info['backend'] = 'kestrel_caffe'

    # generate meta
    scaffold.generate_meta(path, name, 'raven', version)

    # generate parameter
    net_graph = graph.gen_graph(net)
    assert len(net_graph.root) == 1 and len(net_graph.root[0].content.bottom) == 1
    net_info['data'] = net_graph.root[0].content.bottom[0]
    assert len(net_graph.leaf) == 2
    for node in net_graph.leaf:
        assert len(node.content.top) == 1
        if node.content.type == 'Sigmoid':
            if "visibility" in node.content.name:
                net_info['score'] = node.content.top[0]
            else:
                net_info['background'] = node.content.top[0]
        else:
            net_info['point'] = node.content.top[0]

    # adapt for background node
    if not net_info.get('score', None):
        net_info['score'] = net_info['background']
        net_info.pop('background')

    generate_parameter(path, packname, max_batch_size, net_info, cfg_params)

    # compress and save
    scaffold.compress_model(path, [packname], name, version)


@KS_PROCESSOR_REGISTRY.register('raven')
class RavenProcessor(BaseProcessor):
    def process(self):
        version = scaffold.check_version_format(self.version)

        # get param from up config
        with open(self.kestrel_param_json) as f:
            cfg_params = yaml.load(f)
        assert cfg_params['common']['bg']

        net, withBinFile = graph.readNetStructure(self.prototxt, self.model)
        net = scaffold.merge_bn(net)

        process_net(net, cfg_params['structure']['input_h'], cfg_params['structure']
                    ['input_w'], cfg_params['structure'].get('num_kpts', 14))
        generate(net, self.save_path, self.name, self.serialize, self.max_batch_size, cfg_params, version)
