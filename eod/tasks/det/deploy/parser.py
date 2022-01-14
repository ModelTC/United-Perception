import abc
import copy
import yaml
import pandas

import spring.nart.tools.caffe.utils.graph as graph
import spring.nart.tools.kestrel.utils.net_transform as transform

from eod.utils.general.tocaffe_helper import parse_resize_scale


class Parser(abc.ABC):
    def __init__(self, pod_yml_file_or_dict):
        if isinstance(pod_yml_file_or_dict, str):
            with open(pod_yml_file_or_dict, 'r') as f:
                yaml.load(f, Loader=yaml.Loader)
        else:
            self.cfg = pod_yml_file_or_dict

    @abc.abstractmethod
    def get_kestrel_parameters(self):
        """parse training config to get parameters.json for kestrel-sdk
        """
        raise NotImplementedError


class BaseProcessor(object):
    def __init__(self,
                 prototxt,
                 model,
                 v='1.0.0',
                 k='',
                 n='None',
                 p='.',
                 b=8,
                 s=False,
                 resize_hw='',
                 i=3,
                 nnie=False):
        self.prototxt = prototxt
        self.model = model
        self.version = v
        self.kestrel_param_json = k
        self.name = n
        self.save_path = p
        self.max_batch_size = b
        self.serialize = s
        self.resize_hw = resize_hw
        self.input_channel = i
        self.nnie = nnie

    def process(self):
        raise NotImplementedError


def getdotattr(obj, attr_path):
    attr_path = attr_path.split('.')
    for attr in attr_path:
        if attr.isdigit():
            obj = obj[int(attr)]
        elif attr == '[]' or attr == '()':
            continue
        else:
            obj = obj[attr]
    return obj


def get_forground_class_threshes(dataset_cfg, to_kestrel_cfg, with_background_channel=True):
    """Get threshes for each class. The returned list should be the same length with classification channels
    """
    # if kestrel config is given
    if to_kestrel_cfg.get('kestrel_config', None) is not None:
        return to_kestrel_cfg['kestrel_config']

    # process background class
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.update(dataset_cfg.get('train', {}))
    all_class_names = dataset_cfg['dataset']['kwargs']['class_names']
    assert all_class_names[0] == '__background__', f'class_names should include background class'
    # the length of class_names should equal to the number of predicted channels
    forground_class_names = all_class_names[1:]

    assert ('kestrel_config' in to_kestrel_cfg
            or 'metrics_csv' in to_kestrel_cfg
            or 'default_confidence_thresh' in to_kestrel_cfg)

    confidences_thresh = {}
    if to_kestrel_cfg.get('metrics_csv', None) is not None:
        metrics = pandas.read_csv(to_kestrel_cfg['metrics_csv'])
        for idx, row in metrics.iterrows():
            confidences_thresh[row['Class']] = row['FPPI-Score@0.1']
    default_confidence_thresh = to_kestrel_cfg.get('default_confidence_thresh', 0.3)

    # generate new kestrel config
    class_config = [
        {
            # harpy
            'confidence_thresh': confidences_thresh.get(label_name, default_confidence_thresh),
            # essos
            'thresh': confidences_thresh.get(label_name, default_confidence_thresh),
            'id': idx + int(with_background_channel),
            'label': label_name,
            'filter_w': 0,
            'filter_h': 0
        }
        for idx, label_name in enumerate(forground_class_names)
    ]
    return class_config


def add_reshape(net, net_graph, prev_layer, name, reshape_param, insert=True):
    import spring.nart.tools.caffe.convert as convert
    from spring.nart.tools.proto import caffe_pb2 as caffe_pb2
    # prev_node = get_node(graph.gen_graph(net), prev_layer)
    old_top_name = prev_layer.top[0]
    new_top_name = 'reshape_out'

    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Reshape'
    layer.bottom.append(old_top_name)
    layer.top.append(new_top_name)
    layer.reshape_param.shape.dim.extend(reshape_param)
    # if insert:
    #     for succ in prev_node.succ:
    #         succ.content.bottom.remove(old_top_name)
    #         succ.content.bottom.append(new_top_name)
    idx = convert.insertNewLayer(layer, prev_layer.name, net)
    convert.updateNetGraph(net, net_graph)
    return net.layer[idx]


def process_reshape(net, anchor_num, cls_channel_num, anchor_precede=True):
    net_graph = graph.gen_graph(net)
    for node in net_graph.nodes():
        if node.content.type == 'Reshape':
            if (node.content.reshape_param.shape.dim == [0] * 4
                    or len(node.succ) == 0):
                # remove useless reshape
                transform.remove_layer(net, node)
        elif node.content.type == 'Transpose':
            if len(node.succ) == 1 and 'Softmax' == node.succ[0].content.type:
                continue
            # add reshape layer
            reshape_layer = add_reshape(net, net_graph, node.content, 'reshape_anchor_cls', [1, 12, 2, -1], insert=True)
            trans_layer = transform.add_transpose(net, net_graph, reshape_layer, 'anchor_cls', [0, 2, 1, 3])
            print('Add reshape layer: {} + transpose layer: {})'.format(reshape_layer.name, trans_layer.name))
    return net


def parse_dataset_param(dataset_cfg):
    # , class_threshes_file, thresh_name, default_conf_thresh):

    def get_transform(transformer_cfg, type_name):
        for cfg in transformer_cfg:
            if cfg['type'] == type_name:
                return cfg
        return None

    dataset_cfg.update(dataset_cfg.get('test', {}))

    # forground_class_threshes = generate_forground_class_threshes(
    #     dataset_cfg, class_threshes_file, thresh_name, default_conf_thresh)

    kwargs_cfg = getdotattr(dataset_cfg, 'dataset.kwargs')
    # has_keypoint has_mask
    transformer_cfg = kwargs_cfg['transformer']

    # keep scale consistent with tocaffe input blobs shape
    short_scale, long_scale = parse_resize_scale(dataset_cfg)

    pixel_cfg = get_transform(transformer_cfg, 'normalize')
    pixel_means = getdotattr(pixel_cfg, 'kwargs.mean')
    pixel_stds = getdotattr(pixel_cfg, 'kwargs.std')
    color_mode = getdotattr(kwargs_cfg, 'image_reader.kwargs.color_mode')
    assert color_mode in ['RGB', 'GRAY']

    dataset_param = dict()
    # dataset_param['class'] = forground_class_threshes
    dataset_param['short_scale'] = short_scale
    dataset_param['long_scale'] = long_scale
    dataset_param['pixel_means'] = [i * 255 for i in pixel_means]
    dataset_param['pixel_stds'] = [i * 255 for i in pixel_stds]
    dataset_param['rgb_flag'] = color_mode == 'RGB'

    return dataset_param  # , class_meta


def process_sphinx_reshape(net, cls_channel_num, anchor_precede=True, serialize=False):
    net_graph = graph.gen_graph(net)
    for node in net_graph.nodes():
        if node.content.type == 'Reshape':
            if not serialize and (node.content.reshape_param.shape.dim == [0] * 4
                                  or len(node.succ) == 0):
                # remove useless reshape
                transform.remove_layer(net, node)
            elif len(node.succ) == 1 and 'Softmax' == node.succ[0].content.type:
                # update reshape dim
                node.content.reshape_param.shape.ClearField('dim')
                node.content.reshape_param.shape.dim.extend([-1, cls_channel_num, 0, 0])
    return net
