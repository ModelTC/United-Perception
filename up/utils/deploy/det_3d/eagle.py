import os
import json
import yaml
import shutil
import numpy as np
try:
    import spring.nart.tools.kestrel.utils.scaffold as scaffold
    try:
        import spring.nart.tools.proto.caffe_pb2 as caffe_pb2
    except:  # noqa
        from spring.nart.tools.proto import caffe_pb2
except Exception as err:
    print(err)

from up.utils.deploy import parser as up_parser
from up.utils.deploy.parser import BaseProcessor
from up.utils.general.registry_factory import MODEL_HELPER_REGISTRY, KS_PARSER_REGISTRY, KS_PROCESSOR_REGISTRY


@KS_PARSER_REGISTRY.register('eagle')
class EagleParser(up_parser.Parser):
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


def get_forground_class_threshes(dataset_cfg, to_kestrel_cfg, with_background_channel=True):
    """Get threshes for each class. The returned list should be the same length with classification channels
    """
    # if kestrel config is given
    if to_kestrel_cfg.get('kestrel_config', None) is not None:
        return to_kestrel_cfg['kestrel_config']

    forground_class_names = to_kestrel_cfg['class_names']
    confidences_thresh = to_kestrel_cfg.get('confidneces_thresh', {})
    cls_id = {'vehicle': 1420, 'Pedestrian': 221488, 'Cyclist': 1507442}

    # generate new kestrel config
    class_config = [
        {
            # harpy
            'confidence_thresh': confidences_thresh.get(label_name, 0.2),
            # essos
            'thresh': confidences_thresh.get(label_name, 0.2),
            'id': cls_id[label_name],
            'label': label_name,
            'filter_w': 0,
            'filter_h': 0,
            'filter_l': 0
        }
        for idx, label_name in enumerate(forground_class_names)
    ]
    return class_config


def generate_config(train_cfg):
    if isinstance(train_cfg, str):
        with open(train_cfg) as f:
            train_cfg = yaml.load(f)

    kestrel_param = dict()

    # net param
    assert 'net' in train_cfg, 'config file incomplete: lack net infomation'
    model_helper_ins = MODEL_HELPER_REGISTRY[train_cfg.get('model_helper_type', 'base')]
    backbone_model = model_helper_ins(train_cfg['net'][1:3])
    base_anchors_file = backbone_model.roi_head.base_anchors_file

    with open(base_anchors_file, 'r') as f:
        anchor_classes = np.array(json.load(f))

    anchor_size = list()
    cls_order = ['Car', 'Cyclist', 'Pedestrian']
    for cls_name in cls_order:
        for cls_id in range(len(anchor_classes)):
            if anchor_classes[cls_id]['class_name'] == cls_name:
                anchor_size.append(anchor_classes[cls_id]['anchor_sizes'][0])

    anchor_info = {
        'feature_map_stride': anchor_classes[0]['feature_map_stride'],
        'anchor_offset': anchor_classes[0]['anchor_offsets'][0],
        'anchor_stride': anchor_classes[0]['anchor_strides'][0],
        'anchors_size': anchor_size,
        'anchor_rotations': anchor_classes[0]['anchor_rotations']
    }
    kestrel_param.update(anchor_info)

    to_kestrel_cfg = train_cfg['to_kestrel']
    point_cloud_range = to_kestrel_cfg['point_cloud_range']
    start_axis = point_cloud_range[:3]
    end_axis = point_cloud_range[3:6]
    max_voxels_cnt = to_kestrel_cfg.get('max_voxels_cnt', 100000)
    max_points_num = to_kestrel_cfg.get('max_points_num', 100)
    pre_net_maxbatchsize = to_kestrel_cfg.get('pre_net_maxbatchsize', 20000)
    after_top_k = to_kestrel_cfg.get('after_top_k', 1000)
    after_nms_num = to_kestrel_cfg.get('after_nms_num', 300)
    nms_iou_threshold = to_kestrel_cfg.get('nms_iou_threshold', 0.5)
    filter_thresh = to_kestrel_cfg.get('filter_thresh', 0.07)
    voxel_size = to_kestrel_cfg.get('voxel_size', None)

    post_process_info = {
        'start_axis': start_axis, 'end_axis': end_axis,
        'max_voxels_cnt': max_voxels_cnt,
        'max_points_num': max_points_num,
        'pre_net_maxbatchsize': pre_net_maxbatchsize,
        'after_top_k': after_top_k,
        'after_nms_num': after_nms_num,
        'nms_iou_threshold': nms_iou_threshold,
        'filter_thresh': filter_thresh,
        'voxel_size': voxel_size
    }
    kestrel_param.update(post_process_info)

    # threshes for each class
    kestrel_param['class'] = get_forground_class_threshes(train_cfg['dataset'], train_cfg['to_kestrel'])

    return kestrel_param


def generate_common_param(net_info):
    common_param = dict()
    pre_param = {
        "net": net_info['pre']['net_name'],
        "backend": net_info['pre']['backend'],
        "max_batch_size": net_info['pre']['max_batch_size'],
        "input": {
            "data": net_info['pre']['input_name']
        },
        "output": {
            "output": net_info['pre']['output_name']
        }
    }

    det_param = {
        "net": net_info['det']['net_name'],
        "backend": net_info['det']['backend'],
        "max_batch_size": net_info['det']['max_batch_size'],
        "input": {
            "data": net_info['det']['input_name']
        },
        "output": net_info['det']['output_name']
    }

    common_param['pre_net'] = pre_param
    common_param['det_net'] = det_param
    return common_param


@KS_PROCESSOR_REGISTRY.register('eagle')
class EagleProcessor(BaseProcessor):
    def process(self):
        # check meta version format
        version = scaffold.check_version_format(self.version)

        # get param from pod config
        with open(self.kestrel_param_json, 'r') as f:
            kestrel_param = json.load(f)

        net_info = {'pre': dict(), 'det': dict()}
        net_info['pre']['net_name'] = self.model['pfn']['net_name']
        net_info['pre']['input_name'] = self.model['pfn']["input_name"]
        net_info['pre']['output_name'] = list(self.model['pfn']['output_name'])[0]
        net_info['det']['net_name'] = self.model['rpn']['net_name']
        net_info['det']['input_name'] = self.model['rpn']['input_name']
        rpn_output_name = self.model['rpn']['output_name']
        det_output_name = dict()
        for name in rpn_output_name:
            if name.find('reg') >= 0:
                det_output_name['bbox'] = name
            elif name.find('cls') >= 0:
                det_output_name['score'] = name
            elif name.find('dir') >= 0:
                det_output_name['direction'] = name
        net_info['det']['output_name'] = det_output_name

        net_info['pre']['max_batch_size'] = 100000
        net_info['det']['max_batch_size'] = 2
        net_info['pre']['backend'] = 'kestrel_nart'
        net_info['det']['backend'] = 'kestrel_nart'
        net_info['pre']['packname'] = self.model['pfn']['net_name']
        net_info['det']['packname'] = self.model['rpn']['net_name']

        common_param = generate_common_param(net_info)
        kestrel_param['model_files'] = common_param

        self.save_local_path = "tocaffe/" + self.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_local_path)
        # move onnx
        if isinstance(self.model, dict):
            for info in self.model:
                shutil.move("tocaffe/" + self.model[info]['net_name'], self.save_local_path)
        scaffold.generate_json_file(
            os.path.join(self.save_local_path, 'parameters.json'), kestrel_param)

        scaffold.generate_meta(
            self.save_local_path, self.name, 'eagle', version, {'class': kestrel_param['class']})

        scaffold.compress_model(
            self.save_local_path, [net_info['pre']['packname'], net_info['det']['packname']], self.save_path, version)
