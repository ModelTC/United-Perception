import numpy as np
import json
import yaml
import os

try:
    import spring.nart.tools.caffe.convert as convert
    import spring.nart.tools.caffe.utils.graph as graph
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
    try:
        import spring.nart.tools.proto.caffe_pb2 as caffe_pb2
    except:  # noqa
        from spring.nart.tools.proto import caffe_pb2
except Exception as err:
    print(err)

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from up.utils.general.log_helper import default_logger as logger

__all__ = ['ClassifierParser']


@KS_PARSER_REGISTRY.register('classifier')
class ClassifierParser(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def generate_config(train_cfg):
    if isinstance(train_cfg, str):
        with open(train_cfg) as f:
            train_cfg = yaml.load(f)

    kestrel_param = dict()

    kestrel_param['pixel_means'] = train_cfg['to_kestrel'].get('pixel_means', [123.675, 116.28, 103.53])
    kestrel_param['pixel_stds'] = train_cfg['to_kestrel'].get('pixel_stds', [58.395, 57.12, 57.375])
    kestrel_param['is_rgb'] = train_cfg['to_kestrel'].get('is_rgb', True)
    kestrel_param['save_all_label'] = train_cfg['to_kestrel'].get('save_all_label', True)
    kestrel_param['type'] = train_cfg['to_kestrel'].get('type', 'ImageNet')

    if train_cfg.get('to_kestrel') and train_cfg['to_kestrel'].get('class_label'):
        kestrel_param['class_label'] = train_cfg['to_kestrel']['class_label']
    else:
        kestrel_param['class_label'] = {}
        kestrel_param['class_label']['imagenet'] = {}
        kestrel_param['class_label']['imagenet']['calculator'] = 'bypass'
        num_classes = train_cfg.get('num_classes', 1000)
        kestrel_param['class_label']['imagenet']['labels'] = [str(i) for i in np.arange(num_classes)]
        kestrel_param['class_label']['imagenet']['feature_start'] = 0
        kestrel_param['class_label']['imagenet']['feature_end'] = num_classes - 1
    return kestrel_param


def add_reshape(net, net_graph, node, reshape_param, insert=True):

    layer = caffe_pb2.LayerParameter()
    layer.name = node.content.name
    layer.type = 'Reshape'
    layer.bottom.append(node.content.bottom[0])
    layer.top.append(node.content.top[0])
    layer.reshape_param.shape.dim.extend(reshape_param)
    idx = convert.insertNewLayer(layer, node.prev[0], net)
    convert.updateNetGraph(net, net_graph)
    return net.layer[idx]


def process_net(prototxt, model):
    # get net:  net--NetParameter  withBinFile--True
    net, withBinFile = graph.readNetStructure(prototxt, model)  # model-convert.prototxt model-convert.caffemodel
    # process reshape
    net_graph = graph.gen_graph(net)
    for node in net_graph.nodes():
        if node.content.type == 'Reshape':
            if node.content.reshape_param.shape.dim[0] == 1:
                # add new reshape
                tmp = node
                for i in range(5):
                    tmp = tmp.prev[0]
                    if tmp.content.type == 'Convolution':
                        reshape_shape = tmp.content.convolution_param.num_output
                        break
                node.content.reshape_param.shape.ClearField('dim')
                node.content.reshape_param.shape.dim.extend([-1, reshape_shape])
    return net


def generate_category_param(class_label):
    idx = 0
    class_param = dict()
    for category in class_label:
        category_dict = dict()
        category_dict['calculator'] = class_label[category].get('calculator', 'bypass')
        category_dict['feature_start'] = class_label[category].get('start', idx)
        category_dict['feature_end'] = class_label[category].get('end', idx + len(class_label[category]['labels']) - 1)
        category_dict['labels'] = class_label[category]['labels']
        if 'append_dep' in class_label[category].keys():
            category_dict['append_dep'] = class_label[category]['append_dep']
        if 'positive_depend' in class_label[category].keys():
            category_dict['positive_depend'] = class_label[category]['positive_depend']
        if 'proposal_threshold' in class_label[category].keys():
            category_dict['proposal_threshold'] = class_label[category]['proposal_threshold']
        if 'proposal_label_thresholds' in class_label[category].keys():
            category_dict['proposal_label_thresholds'] = class_label[category]['proposal_label_thresholds']
        class_param[category] = category_dict
        idx += len(class_label[category]['labels'])
    return class_param


def generate_parameter(path, packname, max_batch_size, net_info, cfg_params):
    param = dict()
    param['model_files'] = dict()
    net = dict()
    net['net'] = packname
    net['backend'] = net_info['backend']
    net['max_batch_size'] = max_batch_size
    net['input'] = {'data': net_info['data']}
    net['output'] = {'score': net_info['score']}
    param['model_files']['net'] = net
    param['input_h'] = net_info['input_h']
    param['input_w'] = net_info['input_w']
    param['pixel_means'] = cfg_params['pixel_means']
    param['pixel_stds'] = cfg_params['pixel_stds']
    param['is_rgb'] = cfg_params.get('is_rgb', True)
    param['type'] = cfg_params.get('type', 'UNKNOWN')
    param['save_all_label'] = cfg_params.get('save_all_label', True)
    scaffold.generate_json_file(os.path.join(path, 'parameters.json'), param)
    class_label = generate_category_param(cfg_params['class_label'])
    scaffold.generate_json_file(os.path.join(path, 'category_param.json'), class_label)


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
    scaffold.generate_meta(path, name, 'classifier', version)

    # generate parameter

    # get necessary infomation
    net_graph = graph.gen_graph(net)
    # input and output blob
    net_info['data'] = net_graph.root[0].content.bottom[0]
    logger.info(net_graph.root[0].content.name)
    net_graph = graph.gen_graph(net)
    if len(net_graph.leaf) > 1:
        net_info['score'] = [net_graph.leaf[i].content.top[0] for i in range(len(net_graph.leaf))]
        logger.info([net_graph.leaf[i].content.top for i in range(len(net_graph.leaf))])
    else:
        net_info['score'] = net_graph.leaf[0].content.top[0]
        logger.info(net_graph.leaf[0].content.top)

    # input shape
    assert len(net.input_dim) < 1 or len(net.input_shape) < 1
    assert len(net.input_dim) > 0 or len(net.input_shape) > 0
    if len(net.input_dim) > 0:
        assert len(net.input_dim) == 4
        net_info['input_h'] = net.input_dim[2]
        net_info['input_w'] = net.input_dim[3]
    if len(net.input_shape) > 0:
        assert len(net.input_shape) == 1
        assert len(net.input_shape[0].dim) == 4
        net_info['input_h'] = net.input_shape[0].dim[2]
        net_info['input_w'] = net.input_shape[0].dim[3]

    generate_parameter(path, packname, max_batch_size, net_info, cfg_params)

    # compress and save
    scaffold.compress_model(path, [packname, 'category_param.json'], name, version)


@KS_PROCESSOR_REGISTRY.register('classifier')
class ClassifierProcessor(BaseProcessor):
    def process(self):
        version = scaffold.check_version_format(self.version)

        # get param from up config
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        net = process_net(self.prototxt, self.model)
        net = scaffold.merge_bn(net)

        generate(net, self.save_path, self.name, False, 8, kestrel_param, version)
