import os
import json
import yaml
from functools import reduce
import torch

try:
    import spring.nart.tools.caffe.count as count
    import spring.nart.tools.caffe.utils.graph as graph
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
    try:
        import spring.nart.tools.proto.caffe_pb2 as caffe_pb2  # for nart==0.2.4
    except:  # noqa
        from spring.nart.tools.proto import caffe_pb2  # # for nart==1.*.*
except Exception as err:
    print(err)

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY


@KS_PARSER_REGISTRY.register('sphinx')
class SphinxParser(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def get_subnet(net, root):
    bottom_name = root.content.bottom[0]
    spatial_scale = root.content.roi_align_pooling_param.spatial_scale
    nodes = [root]
    subnet = []
    while len(nodes) > 0:
        node = nodes.pop()
        if node.content in net.layer:
            layer = caffe_pb2.LayerParameter()
            layer.CopyFrom(node.content)
            net.layer.remove(node.content)
            subnet.append(layer)
        for succ in node.succ:
            insert = True
            for prev in succ.prev:
                if prev.content in net.layer:
                    insert = False
            if insert:
                nodes.append(succ)
    bottom = {'name': bottom_name, 'spatial_scale': spatial_scale, 'node': root}
    return bottom, subnet

# TODO not support RFCN


def split_net(net, anchor_num, cls_channel_num, bbox_head_type, sample_num, serialize=False):
    net_graph = graph.gen_graph(net)
    net_info = dict()
    rpn_info = dict()
    det_info = dict()

    # parse raw net input
    det_net_root = list()
    det_info['fpn_number'] = 0
    input_roi_names = list()
    for node in net_graph.root:
        rpn_info['data'] = node.content.bottom[0]

    for node in net_graph.nodes():
        if node.content.type == 'ROIAlignPooling':
            input_roi_names.append(node.content.bottom[1])
            det_net_root.append(node)
            det_info['fpn_number'] += 1

    # contruct det net
    det_net = caffe_pb2.NetParameter()
    for i in range(det_info['fpn_number']):
        det_net.input.append(input_roi_names[i])
        det_net.input_dim.extend([1, 5, 1, 1])

    bottoms = list()
    assert len(det_net_root) == det_info['fpn_number']
    for root in det_net_root:
        bottom, subnet = get_subnet(net, root)
        bottoms.append(bottom)
        det_net.layer.extend(subnet)

    def _det_compare_key(element):
        return element['spatial_scale']

    bottoms.sort(key=_det_compare_key, reverse=True)

    assert len(bottoms) == det_info['fpn_number']
    det_info['det_roi_names'] = list()
    det_info['det_feature_names'] = list()
    for i in range(det_info['fpn_number']):
        det_info['det_feature_names'].append(bottoms[i]['name'])
        det_info['det_roi_names'].append(bottoms[i]['node'].content.bottom[1])
        det_net.input.append(det_info['det_feature_names'][-1])

    # replace roialignpooling
    for layer in det_net.layer:
        if layer.type == 'ROIAlign' or layer.type == 'ROIAlignPooling':
            pooled_h = layer.roi_align_pooling_param.pooled_h
            pooled_w = layer.roi_align_pooling_param.pooled_w
            spatial_scale = layer.roi_align_pooling_param.spatial_scale

            layer.ClearField('roi_align_pooling_param')
            layer.type = 'PODROIAlignPooling'
            layer.podroi_align_pooling_param.pooled_h = pooled_h
            layer.podroi_align_pooling_param.pooled_w = pooled_w
            layer.podroi_align_pooling_param.spatial_scale = spatial_scale
            layer.podroi_align_pooling_param.sample_num = sample_num

    # processed net -> rpn_net
    rpn_net = net

    # process reshape
    up_parser.process_sphinx_reshape(rpn_net, 2, anchor_precede=False, serialize=serialize)
    up_parser.process_reshape(det_net, 1, cls_channel_num, anchor_precede=False)

    # infer rpn net info
    rpn_info['rpn_bbox_names'] = list()
    rpn_info['rpn_score_names'] = list()
    _, _, rpn_shape = count.inferNet(rpn_net)
    for i in range(det_info['fpn_number']):
        det_net.input_dim.extend(rpn_shape[det_info['det_feature_names'][i]])
    rpn_graph = graph.gen_graph(rpn_net)
    # TODO
    # assert len(rpn_graph.leaf) == 2 *

    # process nninterp
    for node in rpn_graph.nodes():
        if node.content.type == 'NNInterp':
            if node.content.nninterp_param.HasField('height'):
                node.content.nninterp_param.ClearField('height')
                node.content.nninterp_param.ClearField('width')
                assert len(node.succ) == 1
                succ_node = node.succ[0]
                assert succ_node.content.type == 'Eltwise'
                assert len(succ_node.prev) == 2
                target_node = succ_node.prev[0] if succ_node.prev[1] == node \
                    else succ_node.prev[1]
                assert len(target_node.content.top) == 1
                node.content.bottom.append(target_node.content.top[0])

    rpn_bbox_blob_info = list()
    rpn_score_blob_info = list()
    for leaf in rpn_graph.leaf:
        if not leaf.content.top[0] in det_info['det_feature_names']:
            shape = rpn_shape[leaf.content.top[0]]
            if shape[1] == anchor_num * 4:
                rpn_bbox_blob_info.append({'name': leaf.content.top[0],
                                           'node': leaf, 'elem_count': reduce(lambda x, y: x * y, shape)})
            else:
                rpn_score_blob_info.append({'name': leaf.content.top[0],
                                            'node': leaf, 'elem_count': reduce(lambda x, y: x * y, shape)})

    assert len(rpn_bbox_blob_info) == len(rpn_score_blob_info)

    def _rpn_compare_key(element):
        return element['elem_count']

    def _find_leaf_nodes(root_node):
        leaf_nodes = []
        if len(root_node.succ) == 0:
            return [root_node]
        for node in root_node.succ:
            leaf_nodes.extend(_find_leaf_nodes(node))
        return leaf_nodes

    def _same_pair(score_node, bbox_node):
        assert len(bbox_node.prev) == 1
        root_node = bbox_node.prev[0]
        neighbor_node = root_node.succ[0] if root_node.succ[1] == bbox_node \
            else root_node.succ[1]
        return score_node in _find_leaf_nodes(neighbor_node)

    rpn_bbox_blob_info.sort(key=_rpn_compare_key, reverse=True)
    for bbox_blob in rpn_bbox_blob_info:
        rpn_info['rpn_bbox_names'].append(bbox_blob['name'])
        for score_blob in rpn_score_blob_info:
            if _same_pair(score_blob['node'], bbox_blob['node']):
                rpn_info['rpn_score_names'].append(score_blob['name'])
                break

    # inference det net info
    _, _, det_shape = count.inferNet(det_net)
    det_graph = graph.gen_graph(det_net)
    assert len(det_graph.leaf) == 2
    for leaf in det_graph.leaf:
        shape = det_shape[leaf.content.top[0]]
        if shape[1] == cls_channel_num:
            det_info['score'] = leaf.content.top[0]
        else:
            det_info['bbox'] = leaf.content.top[0]

    net_info['rpn'] = rpn_info
    net_info['det'] = det_info
    return rpn_net, det_net, net_info


def generate_config(train_cfg):
    if isinstance(train_cfg, str):
        with open(train_cfg) as f:
            train_cfg = yaml.load(f)

    kestrel_param = dict()
    # net param
    assert 'net' in train_cfg, 'config file incomplete: lack net infomation'
    model_helper_ins = MODEL_HELPER_REGISTRY[train_cfg.get('model_helper_type', 'base')]
    model = model_helper_ins(train_cfg['net'])

    for mname in ['backbone', 'neck', 'roi_head', 'bbox_head']:
        assert hasattr(model, mname)

    kestrel_net_param = dict()
    if hasattr(model, 'neck'):
        strides = model.neck.get_outstrides()
    else:
        strides = model.backbone.get_outstrides()

    if torch.is_tensor(strides):
        strides = strides.tolist()
    if not hasattr(model, 'post_process') and hasattr(model, 'roi_head'):
        setattr(model, 'post_process', model.roi_head)
    kestrel_param['with_background_rpn'] = False
    kestrel_param['with_background_head'] = False
    if model.post_process.cls_loss.activation_type == 'softmax':
        kestrel_param['with_background_rpn'] = True

    model.post_process.anchor_generator.build_base_anchors(strides)
    kestrel_anchors = model.post_process.anchor_generator.export()
    kestrel_net_param.update(kestrel_anchors)

    # net param
    kestrel_net_param['detection_strides'] = strides
    kestrel_net_param['fpn_output_num'] = len(kestrel_net_param['detection_strides'])

    # rpn
    predictor = model.post_process.test_predictor
    kestrel_net_param['num_anchors'] = model.post_process.num_anchors
    # sphinx only use anchor_ratios and anchor_scale to compute num_anchors
    # TODO: help sphinx deprecate anchor_ratios && anchor_scales
    kestrel_net_param['anchor_ratios'] = [-1]
    kestrel_net_param['anchor_scales'] = [-1] * model.post_process.num_anchors

    kestrel_net_param['pre_top_k'] = predictor.pre_nms_top_n
    kestrel_net_param['aft_top_k'] = predictor.post_nms_top_n
    kestrel_net_param['rpn_nms_thresh'] = predictor.nms_cfg['nms_iou_thresh']
    kestrel_net_param['roi_min_size'] = predictor.roi_min_size
    kestrel_net_param['top_n_across_levels'] = predictor.merger.top_n
    kestrel_net_param['rpn_bbox_score_thresh'] = predictor.pre_nms_score_thresh

    # bbox_head
    if model.bbox_head.cls_loss.activation_type == 'softmax':
        kestrel_param['with_background_head'] = True
    predictor = model.bbox_head.predictor
    kestrel_net_param['det_nms_thresh'] = predictor.nms_cfg['nms_iou_thresh']
    kestrel_net_param['share_location'] = predictor.share_location
    kestrel_net_param['detect_top_k'] = predictor.top_n
    kestrel_net_param['det_bbox_score_thresh'] = predictor.bbox_score_thresh
    kestrel_net_param['base_scale'] = model.bbox_head.cfg['fpn']['base_scale']
    kestrel_net_param['bbox_head_fpn_levels'] = model.bbox_head.cfg['fpn']['fpn_levels']
    kestrel_net_param['backbone_output_num'] = len(kestrel_net_param['bbox_head_fpn_levels'])
    kestrel_net_param['roipooling_sample_num'] = model.bbox_head.cfg['roipooling']['sampling_ratio']
    kestrel_param.update(kestrel_net_param)

    # dataset param
    assert 'dataset' in train_cfg, 'config file incomplete: lack dataset'
    dataset_param = up_parser.parse_dataset_param(train_cfg['dataset'])
    kestrel_param.update(dataset_param)

    # threshes for each class
    kestrel_param['class'] = up_parser.get_forground_class_threshes(train_cfg['dataset'], train_cfg['to_kestrel'])

    kestrel_param['bbox_head_type'] = model.bbox_head.__class__.__name__

    return kestrel_param


def process_net(prototxt, model, anchor_num, cls_channel_num, input_h, input_w,
                bbox_head_type, sample_num, serialize, input_channel=3):
    # get net
    net, withBinFile = graph.readNetStructure(prototxt, model)

    # update input dim
    scaffold.update_input_dim(net, 0, [1, input_channel, input_h, input_w])

    # parse and update
    # split rpn & det and get input/output info
    rpn_net, det_net, net_info = split_net(net, anchor_num, cls_channel_num,
                                           bbox_head_type, sample_num, serialize)

    # merge bn
    rpn_net = scaffold.merge_bn(rpn_net)
    det_net = scaffold.merge_bn(det_net)

    return rpn_net, det_net, net_info


def get_rpn_cls_softmax_layer(net):
    top_names = []
    softmax_layers = []
    net_graph = graph.gen_graph(net)
    for node in net_graph.nodes():
        if node.content.type == 'Reshape':
            if len(node.succ) == 0:
                top_names += node.content.bottom
    for node in net_graph.nodes():
        if node.content.type == 'Softmax':
            if node.content.top[0] in top_names:
                softmax_layers.append(node.content.name)
    return softmax_layers


def generate_common_param(net_info, kestrel_param, max_batch_size):
    assert len(kestrel_param['detection_strides']) == len(net_info['rpn']['rpn_bbox_names'])
    common_param = dict()
    rpn_param = dict()
    det_param = dict()
    rpn_param['net'] = net_info['rpn']['packname']
    rpn_param['backend'] = net_info['rpn']['backend']
    rpn_param['max_batch_size'] = max_batch_size
    rpn_param['input'] = {'data': net_info['rpn']['data']}
    rpn_param['output'] = dict()
    rpn_param['marked_output'] = dict()
    for i in range(len(kestrel_param['detection_strides'])):
        rpn_param['output']['bbox_' + str(i)] = net_info['rpn']['rpn_bbox_names'][i]
        rpn_param['output']['score_' + str(i)] = net_info['rpn']['rpn_score_names'][i]
    for i in range(net_info['det']['fpn_number']):
        rpn_param['marked_output']['shared_rpn_layer_' + str(i)] = \
            net_info['det']['det_feature_names'][i]

    det_param['net'] = net_info['det']['packname']
    det_param['backend'] = net_info['det']['backend']
    det_param['max_batch_size'] = max_batch_size * kestrel_param['aft_top_k']
    det_param['input'] = dict()
    for i in range(net_info['det']['fpn_number']):
        det_param['input']['feature_' + str(i)] = net_info['det']['det_feature_names'][i]
        det_param['input']['roi_' + str(i)] = net_info['det']['det_roi_names'][i]
    det_param['output'] = {'bbox': net_info['det']['bbox'], 'score': net_info['det']['score']}

    common_param['rpn_net'] = rpn_param
    common_param['det_net'] = det_param
    return common_param


@KS_PROCESSOR_REGISTRY.register('sphinx')
class SphinxProcessor(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        # get resize hw if provided
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['short_scale'] = h
            kestrel_param['long_scale'] = w

        # process net
        # channel number for classification
        if kestrel_param.get('sigmoid', False):
            offset = 0
        else:
            offset = 1
        cls_channel_num = len(kestrel_param['class']) + offset
        rpn_net, det_net, net_info = process_net(self.prototxt,
                                                 self.model,
                                                 kestrel_param['num_anchors'],
                                                 cls_channel_num,
                                                 kestrel_param['short_scale'],
                                                 kestrel_param['long_scale'],
                                                 kestrel_param['bbox_head_type'],
                                                 kestrel_param['roipooling_sample_num'],
                                                 self.serialize,
                                                 self.input_channel)
        assert len(kestrel_param['bbox_head_fpn_levels']) == net_info['det']['fpn_number']

        net_info['rpn']['packname'] = 'proposal'
        net_info['det']['packname'] = 'detection'
        net_info['rpn']['backend'] = 'kestrel_caffe'
        net_info['det']['backend'] = 'kestrel_caffe'
        rpn_path = scaffold.generate_model(rpn_net, self.save_path, net_info['rpn']['packname'])
        scaffold.generate_model(det_net, self.save_path, net_info['det']['packname'])

        if self.serialize:
            rpn_cls_softmax_layer = get_rpn_cls_softmax_layer(rpn_net)
            net_info['rpn']['packname'] = 'rpn_engine.bin'
            net_info['rpn']['backend'] = 'kestrel_mixnet'
            engine_path = os.path.join(self.save_path, net_info['rpn']['packname'])
            config = {'output_names': net_info['det']['det_feature_names'], 'set_caffe_layers': rpn_cls_softmax_layer}
            scaffold.serialize(rpn_path, self.max_batch_size, engine_path, config)

        common_param = generate_common_param(net_info, kestrel_param, self.max_batch_size)
        kestrel_param['model_files'] = common_param

        scaffold.generate_json_file(os.path.join(self.save_path, 'parameters.json'), kestrel_param)

        scaffold.generate_meta(self.save_path, self.name, 'sphinx', version,
                               {'class': kestrel_param['class']})

        scaffold.compress_model(self.save_path, [net_info['rpn']['packname'], net_info['det']['packname']],
                                self.save_path, version)
