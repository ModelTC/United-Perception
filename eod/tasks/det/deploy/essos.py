import os
import yaml
import json
import torch

import spring.nart.tools.caffe.count as count
import spring.nart.tools.caffe.utils.graph as graph
import spring.nart.tools.kestrel.utils.scaffold as scaffold

from . import parser as eod_parser
from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY, KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY
from .parser import BaseProcessor

__all__ = ['EssosParser', 'EssosProcessor']


@KS_PARSER_REGISTRY.register('essos')
class EssosParser(eod_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def generate_config(train_cfg):
    if isinstance(train_cfg, str):
        with open(train_cfg) as f:
            train_cfg = yaml.load(f)

    kestrel_param = dict()

    # parse config parameters needed
    assert 'net' in train_cfg, 'config file incomplete: lack net infomation'
    model_helper_ins = MODEL_HELPER_REGISTRY[train_cfg.get('model_helper_type', 'base')]
    model = model_helper_ins(train_cfg['net'])
    for mname in ['backbone', 'roi_head']:
        assert hasattr(model, mname)
    for mname in ['bbox_head', 'mask_head', 'keyp_head']:
        assert not hasattr(model, mname)

    kestrel_net_param = dict()
    # with open(anchor_config, 'r') as f:
    #     kestrel_net_param['anchors'] = json.load(f)

    if hasattr(model, 'neck'):
        strides = model.neck.get_outstrides()
    else:
        strides = model.backbone.get_outstrides()

    # model.roi_head.anchor_generator.build_base_anchors(strides)
    if torch.is_tensor(strides):
        strides = strides.tolist()
    model.post_process.anchor_generator.build_base_anchors(strides)
    # kestrel_anchors = model.roi_head.anchor_generator.export()
    kestrel_anchors = model.post_process.anchor_generator.export()
    kestrel_net_param.update(kestrel_anchors)
    # kestrel_net_param['anchors'] = model.roi_head.anchor_generator.export()

    kestrel_net_param['rpn_stride'] = strides
    # kestrel_net_param['num_anchors'] = model.roi_head.num_anchors
    kestrel_net_param['num_anchors'] = model.post_process.num_anchors
    # topk = model.roi_head.test_predictor.post_nms_top_n
    topk = model.post_process.test_predictor.post_nms_top_n
    kestrel_net_param['rpn_top_n'] = [topk] * len(strides)
    # kestrel_net_param['aft_top_k'] = model.roi_head.test_predictor.merger.top_n
    kestrel_net_param['aft_top_k'] = model.post_process.test_predictor.merger.top_n
    # kestrel_net_param['nms_thresh'] = model.roi_head.test_predictor.merger.nms_cfg['nms_iou_thresh']
    kestrel_net_param['nms_thresh'] = model.post_process.test_predictor.merger.nms_cfg.get('nms_iou_thresh', 0.5)
    # kestrel_net_param['with_background_class'] = model.roi_head.with_background_channel
    kestrel_net_param['with_background_class'] = model.post_process.with_background_channel

    kestrel_param.update(kestrel_net_param)

    # dataset param
    assert 'dataset' in train_cfg, 'config file incomplete: lack dataset'
    dataset_param = eod_parser.parse_dataset_param(train_cfg['dataset'])
    # train_cfg['dataset'], kestrel_config, thresh_name='thresh', default_conf_thresh=0)
    kestrel_param.update(dataset_param)

    # threshes for each class
    kestrel_param['class'] = eod_parser.get_forground_class_threshes(
        train_cfg['dataset'], train_cfg['to_kestrel'],
        with_background_channel=model.post_process.with_background_channel)

    return kestrel_param


def is_score_branch(node):
    typ = node.content.type
    parent_typ = '' if len(node.prev) != 1 else node.prev[0].content.type
    return typ == 'Sigmoid' or typ == 'Softmax' or parent_typ == 'Sigmoid' or parent_typ == 'Softmax'


def process_net(prototxt, model, anchor_num, cls_channel_num, input_h, input_w, input_channel=3):
    # get net
    net, withBinFile = graph.readNetStructure(prototxt, model)

    # update input dim
    scaffold.update_input_dim(net, 0, [1, input_channel, input_h, input_w])

    # merge bn
    net = scaffold.merge_bn(net)

    # process reshape
    eod_parser.process_reshape(net, anchor_num, cls_channel_num, anchor_precede=True)

    # get net info
    net_info = dict()
    net_graph = graph.gen_graph(net)

    assert len(net_graph.root) == 1
    net_info['data'] = net_graph.root[0].content.bottom[0]

    # select score and bbox output
    score = list()
    bbox = list()
    for leaf in net_graph.leaf:
        if is_score_branch(leaf):
            score.append(leaf.content.top[0])
        else:
            bbox.append(leaf.content.top[0])

    _, _, blob_shape = count.inferNet(net)

    score.sort(key=lambda x: blob_shape[x][3], reverse=True)
    bbox.sort(key=lambda x: blob_shape[x][3], reverse=True)

    net_info['output'] = list()
    for i in range(len(bbox)):
        net_info['output'].append([score[i], bbox[i]])

    return net, net_info


def generate_common_param(net_info, max_batch_size):
    common_param = dict()
    net_param = dict()
    net_param['net'] = net_info['packname']
    net_param['backend'] = net_info['backend']
    net_param['max_batch_size'] = max_batch_size
    net_param['input'] = {'data': net_info['data']}

    net_param['output'] = dict()
    rpn_blob_param = list()
    for i, pair in enumerate(net_info['output']):
        score_key = 'score' + str(i)
        bbox_key = 'bbox' + str(i)
        net_param['output'][score_key] = pair[0]
        net_param['output'][bbox_key] = pair[1]
        rpn_blob_param.append([score_key, bbox_key])

    common_param['net'] = net_param
    return common_param, rpn_blob_param


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
        cls_channel_num = len(kestrel_param['class']) + int(kestrel_param['with_background_class'])
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['short_scale'] = h
            kestrel_param['long_scale'] = w

        net, net_info = process_net(self.prototxt, self.model, kestrel_param['num_anchors'], cls_channel_num,
                                    kestrel_param['short_scale'], kestrel_param['long_scale'], self.input_channel)

        net_info['packname'] = 'model'
        net_info['backend'] = 'kestrel_caffe'
        # save model
        model_path = scaffold.generate_model(net, self.save_path, net_info['packname'])

        # serialize
        if self.serialize:
            net_info['packname'] = 'engine.bin'
            net_info['backend'] = 'kestrel_mixnet'
            engine_path = os.path.join(self.save_path, net_info['packname'])
            scaffold.serialize(model_path, self.max_batch_size, engine_path)
        if self.nnie:
            if self.input_channel == 1:
                kestrel_param['pixel_means'] = [0] * self.input_channel
                kestrel_param['pixel_stds'] = [1] * self.input_channel
            net_info['packname'] = 'engine.bin'
            net_info['backend'] = 'kestrel_nart'
            net_info['nnie_cfg'] = 'engine.bin.json'

        common_param, rpn_blob_param = generate_common_param(net_info, self.max_batch_size)
        kestrel_param['model_files'] = common_param
        kestrel_param['rpn_blobs'] = rpn_blob_param

        scaffold.generate_json_file(os.path.join(self.save_path, 'parameters.json'), kestrel_param)
        scaffold.generate_meta(self.save_path, self.name, 'essos', version, {'class': kestrel_param['class']})
        pack_list = [net_info['packname']]
        if self.nnie:
            pack_list.append(net_info['nnie_cfg'])
        scaffold.compress_model(self.save_path, pack_list, self.save_path, version)
