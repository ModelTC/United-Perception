import os
import json
import yaml

try:
    import spring.nart.tools.caffe.count as count
    import spring.nart.tools.caffe.utils.graph as graph
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
    try:
        import spring.nart.tools.proto.caffe_pb2 as caffe_pb2
    except:  # noqa
        from spring.nart.tools.proto import caffe_pb2
except Exception as err:
    print(err)

from . import parser as up_parser
from .parser import BaseProcessor
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY


@KS_PARSER_REGISTRY.register('harpy')
class HarpyParser(up_parser.Parser):
    def get_kestrel_parameters(self):
        return generate_config(self.cfg)


def get_subnet(net, root):
    bottom = root.content.bottom[0]
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
    return bottom, subnet


def split_net(net, anchor_num, cls_channel_num, bbox_head_type):
    net_graph = graph.gen_graph(net)
    net_info = dict()
    rpn_info = dict()
    det_info = dict()
    roi_blob_name = 'roi'  # specify roi input name

    # parse raw net input: 2 dummy input and 1 rpn input
    det_net_root = []  # save two det net Convolution root layer
    for node in net_graph.root:
        if node.content.type == 'DummyData':
            det_info['roi'] = roi_blob_name
            node.succ[0].content.bottom.remove(node.content.top[0])
            node.succ[0].content.bottom.append(roi_blob_name)
            if 'RFCN' in bbox_head_type:
                for brother in node.succ[0].prev:
                    if brother.content.type != 'DummyData':
                        det_net_root.append(brother)
            else:
                det_net_root.append(node.succ[0])
            net.layer.remove(node.content)
        else:
            rpn_info['data'] = node.content.bottom[0]

    # construct det net
    det_net = caffe_pb2.NetParameter()
    det_net.input.append(roi_blob_name)
    det_net.input_dim.extend([1, 5, 1, 1])

    subnet_bottom = set()
    assert len(det_net_root) == 2 if 'RFCN' in bbox_head_type else 1
    for root in det_net_root:
        bottom, subnet = get_subnet(net, root)
        subnet_bottom.add(bottom)
        det_net.layer.extend(subnet)

    # add feature input for det net
    assert len(subnet_bottom) == 1
    feature_blob_name = subnet_bottom.pop()
    det_net.input.append(feature_blob_name)
    rpn_info['feature'] = feature_blob_name

    # processed net -> rpn_net
    rpn_net = net

    # process reshape
    up_parser.process_reshape(rpn_net, anchor_num, 2, anchor_precede=False)
    up_parser.process_reshape(det_net, 1, cls_channel_num, anchor_precede=False)

    # inference rpn net info
    _, _, rpn_shape = count.inferNet(rpn_net)
    det_net.input_dim.extend(rpn_shape[feature_blob_name])
    rpn_graph = graph.gen_graph(rpn_net)
    assert len(rpn_graph.leaf) == 3 if 'RFCN' in bbox_head_type else 2
    for leaf in rpn_graph.leaf:
        if feature_blob_name != leaf.content.top[0]:
            shape = rpn_shape[leaf.content.top[0]]
            if shape[1] == anchor_num * 4:
                rpn_info['bbox'] = leaf.content.top[0]
            else:
                rpn_info['score'] = leaf.content.top[0]

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

    for mname in ['backbone', 'roi_head', 'bbox_head']:
        assert hasattr(model, mname)
    for mname in ['neck']:
        assert not hasattr(model, mname)

    kestrel_net_param = dict()
    strides = model.backbone.get_outstrides()
    assert len(strides) == 1, strides
    model.post_process.anchor_generator.build_base_anchors(strides)
    kestrel_anchors = model.post_process.anchor_generator.export()
    kestrel_net_param.update(kestrel_anchors)
    kestrel_net_param['anchors'] = kestrel_net_param['anchors'][0]

    kestrel_net_param['detection_stride'] = strides[0]
    kestrel_net_param['num_anchors'] = model.post_process.num_anchors
    # sphinx only use anchor_ratios and anchor_scale to compute num_anchors
    # TODO: help sphinx deprecate anchor_ratios && anchor_scales
    kestrel_net_param['anchor_ratios'] = [-1]
    kestrel_net_param['anchor_scales'] = [-1] * model.post_process.num_anchors

    kestrel_net_param['pre_top_k'] = model.post_process.test_predictor.pre_nms_top_n
    kestrel_net_param['aft_top_k'] = model.post_process.test_predictor.post_nms_top_n
    kestrel_net_param['rpn_nms_thresh'] = model.post_process.test_predictor.nms_cfg['nms_iou_thresh']
    kestrel_net_param['det_nms_thresh'] = model.bbox_head.predictor.nms_cfg['nms_iou_thresh']

    kestrel_param.update(kestrel_net_param)

    # dataset param
    assert 'dataset' in train_cfg, 'config file incomplete: lack dataset'
    dataset_param = up_parser.parse_dataset_param(train_cfg['dataset'])

    kestrel_param.update(dataset_param)

    # parse parameters for softer nms
    for temp in train_cfg['net']:
        if temp['name'] == 'bbox_head':
            nms_cfg = temp['kwargs']['cfg']['bbox_predictor']['kwargs']['nms']
            if nms_cfg['type'] == 'soft':
                soft_nms_cfg = {}
                soft_nms_cfg['use_softnms'] = True
                soft_nms_cfg['softnms_sigma'] = nms_cfg.get("softnms_sigma", 0.5)
                soft_nms_cfg['softnms_nt'] = nms_cfg.get('nms_isou_thresh')
                soft_nms_cfg['softnms_thresh'] = nms_cfg.get('softnms_bbox_score_thresh', 0.0001)
                soft_nms_cfg['softnms_method'] = nms_cfg.get('softnms_method', 'linear')
                kestrel_net_param.update(soft_nms_cfg)
                break

    # threshes for each class
    kestrel_param['class'] = up_parser.get_forground_class_threshes(train_cfg['dataset'], train_cfg['to_kestrel'])

    kestrel_param['bbox_head_type'] = model.bbox_head.__class__.__name__

    return kestrel_param


def process_net(prototxt, model, anchor_num, cls_channel_num, input_h, input_w, bbox_head_type, input_channel=3):
    # get net
    net, withBinFile = graph.readNetStructure(prototxt, model)

    # update input dim
    scaffold.update_input_dim(net, 0, [1, input_channel, input_h, input_w])

    # process reshape
    # pod.process_reshape(net, anchor_num, cls_channel_num, anchor_precede=False)

    # parse and update
    # split rpn & det and get input/output info
    rpn_net, det_net, net_info = split_net(net, anchor_num, cls_channel_num, bbox_head_type)

    # merge bn
    rpn_net = scaffold.merge_bn(rpn_net)
    det_net = scaffold.merge_bn(det_net)

    return rpn_net, det_net, net_info


def generate_common_param(net_info, max_batch_size, aft_top_k):
    common_param = dict()
    rpn_param = dict()
    det_param = dict()
    rpn_param['net'] = net_info['rpn']['packname']
    rpn_param['backend'] = net_info['rpn']['backend']
    rpn_param['max_batch_size'] = max_batch_size
    rpn_param['input'] = {'data': net_info['rpn']['data']}
    rpn_param['output'] = {'bbox': net_info['rpn']['bbox'], 'score': net_info['rpn']['score']}
    rpn_param['marked_output'] = {'shared_rpn_layer': net_info['rpn']['feature']}

    det_param['net'] = net_info['det']['packname']
    det_param['backend'] = net_info['det']['backend']
    det_param['max_batch_size'] = max_batch_size * aft_top_k
    det_param['input'] = {'feature': net_info['rpn']['feature'], 'roi': net_info['det']['roi']}
    det_param['output'] = {'bbox': net_info['det']['bbox'], 'score': net_info['det']['score']}

    common_param['rpn_net'] = rpn_param
    common_param['det_net'] = det_param
    return common_param


@KS_PROCESSOR_REGISTRY.register('harpy')
class HarpyProcessor(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)

        # get param from pod config
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        # get resize hw if provided
        if self.resize_hw != '':
            h, w = [int(i) for i in self.resize_hw.strip().split("x")]
            kestrel_param['short_scale'] = h
            kestrel_param['long_scale'] = w
        # process_net
        # number of channel for classification
        if kestrel_param.get('sigmoid', False):
            offset = 0
        else:
            offset = 1
        cls_channel_num = len(kestrel_param['class']) + offset
        print('cls_channel_num:{}'.format(cls_channel_num))
        rpn_net, det_net, net_info = process_net(self.prototxt,
                                                 self.model,
                                                 kestrel_param['num_anchors'],
                                                 cls_channel_num,
                                                 kestrel_param['short_scale'],
                                                 kestrel_param['long_scale'],
                                                 kestrel_param['bbox_head_type'],
                                                 self.input_channel)

        net_info['rpn']['packname'] = 'proposal'
        net_info['det']['packname'] = 'detection'
        net_info['rpn']['backend'] = 'kestrel_caffe'
        net_info['det']['backend'] = 'kestrel_caffe'
        rpn_path = scaffold.generate_model(rpn_net, self.save_path, net_info['rpn']['packname'])
        scaffold.generate_model(det_net, self.save_path, net_info['det']['packname'])

        if self.serialize:
            net_info['rpn']['packname'] = 'rpn_engine.bin'
            net_info['rpn']['backend'] = 'kestrel_mixnet'
            engine_path = os.path.join(self.save_path, net_info['rpn']['packname'])
            config = {'output_names': [net_info['rpn']['feature']]}
            scaffold.serialize(rpn_path, self.max_batch_size, engine_path, config)

        common_param = generate_common_param(net_info, self.max_batch_size,
                                             kestrel_param['aft_top_k'])
        kestrel_param['model_files'] = common_param

        scaffold.generate_json_file(
            os.path.join(self.save_path, 'parameters.json'), kestrel_param)

        scaffold.generate_meta(
            self.save_path, self.name, 'harpy', version, {'class': kestrel_param['class']})

        scaffold.compress_model(
            self.save_path, [net_info['rpn']['packname'], net_info['det']['packname']], self.save_path, version)
