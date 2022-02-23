import yaml
import re
import os.path
import json
import copy

from .user_analysis_helper import get_task_from_cfg
from .log_helper import default_logger as logger


class IncludeLoader(yaml.Loader):
    _cache = {}

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(IncludeLoader, self).__init__(stream)

        self.overwrite_keys = set(['overwrite@', 'override@'])

    def _split(self, path):
        splits = path.split('//', 1)
        if len(splits) == 1:
            return splits

        path, split = splits

        splits = re.split(r'//|\.', split)

        return [path] + splits

    # include
    def include(self, node):
        # splits = node.value.split('//')
        splits = self._split(node.value)
        node.value = splits[0]

        v = self.extract_file(self.construct_scalar(node))

        for k in splits[1:]:
            if isinstance(v, list):
                v = v[int(k)]
            else:
                v = v[k]
        return v

    # include, overwrite api
    def extract_file(self, filename):
        path = os.path.join(self._root, filename)

        if path in IncludeLoader._cache:
            return IncludeLoader._cache[path]

        # with open(path, 'r') as f:
        #     v = yaml.load(f, IncludeLoader)

        v = load_yaml(path)

        IncludeLoader._cache[path] = v
        return v

    # overwrite
    def _process_overwrite(self, data, k):
        def key2tuple(key):
            current = data
            lst = []
            for k in key.split('.'):
                if isinstance(current, list):
                    k = int(k)
                try:
                    current = current[k]
                except:  # noqa
                    current = {}
                lst.append(k)
            return tuple(lst)

        def replace(overwrite_key, v):
            current = data
            keys = overwrite_key.split('.')

            # non-recursive
            for idx, k in enumerate(keys):
                if isinstance(current, dict):
                    if idx == len(keys) - 1:
                        current[k] = v
                        return

                    if k in current:
                        current = current[k]
                    else:
                        current[k] = {}
                        current = current[k]
                elif isinstance(current, list):
                    k = int(k)
                    assert k <= len(current)
                    if idx != len(keys) - 1:
                        if k < len(current):
                            current = current[k]
                        else:
                            current.append({})
                            current = current[k]
                    else:
                        if k < len(current):
                            current[k] = v
                        else:
                            current.append(v)
                else:
                    raise Exception(f'invalid replacement "{overwrite_key}" to {v}')

        overwrites = data[k]
        for overwrite_key, v in sorted(overwrites.items(), key=lambda x: key2tuple(x[0])):
            replace(overwrite_key, v)

        del data[k]
        return data

    # for overwrite
    def construct_document(self, node):
        data = super(IncludeLoader, self).construct_document(node)
        for k in self.overwrite_keys:
            if k in data:
                self._process_overwrite(data, k)
        return data


def check_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    task_type = get_task_from_cfg(cfg)
    cfg['saver']['task_type'] = task_type
    if task_type == 'det':
        net = cfg.get('net', [])
        assert len(net) > 0, "net doesn't exist."
        net_parse, idx2name = {}, {}
        num_level = 0
        for idx in range(len(net)):
            name = net[idx]['name']
            net_parse[name] = net[idx]
            if name == 'neck':
                num_level = net_parse[name]['kwargs'].get('num_level', 0)
            idx2name[name] = idx
        # NaiveRPN: num_classes
        roi_head_type = net_parse['roi_head']['type']
        if roi_head_type == 'NaiveRPN':
            logger.info("auto reset rpn num_classes = 2")
            cfg['net'][idx2name['roi_head']]['kwargs']['num_classes'] = 2
            cfg['net'][idx2name['post_process']]['kwargs']['num_classes'] = 2
        # roi_head: class_activation
        name_post = 'yolox_post' if 'yolox_post' in net_parse else 'post_process'
        cls_loss_type = net_parse[name_post]['kwargs']['cfg']['cls_loss']['type']
        class_activation = 'softmax' if 'softmax' in cls_loss_type else 'sigmoid'
        logger.info('auto reset class activation')
        cfg['net'][idx2name['roi_head']]['kwargs']['class_activation'] = class_activation
        # roi_head: num_level:
        if num_level != 0:
            cfg['net'][idx2name['roi_head']]['kwargs']['num_level'] = num_level
        elif roi_head_type != 'YoloXHead':
            cfg['net'][idx2name['roi_head']]['kwargs']['num_level'] = 1
        # roi_head: num_anchors or num_point
        anchors = cfg['net'][idx2name[name_post]]['kwargs']['cfg']['anchor_generator']
        anchor_ratios = anchors['kwargs'].get('anchor_ratios', [])
        anchor_scales = anchors['kwargs'].get('anchor_scales', [])
        num_anchors = len(anchor_ratios) * len(anchor_scales)
        logger.info('auto reset num_anchors or num_point base on anchor generator')
        if roi_head_type not in ['YoloXHead']:
            cfg['net'][idx2name['roi_head']]['kwargs']['num_anchors'] = num_anchors
        else:
            cfg['net'][idx2name['roi_head']]['kwargs']['num_point'] = num_anchors if num_anchors > 0 else 1
    else:
        pass

    return cfg


IncludeLoader.add_constructor('!include', IncludeLoader.include)


def load_yaml(path, cfg_type='up'):
    with open(path, "r")as f:
        yaml_data = yaml.load(f, IncludeLoader)
    if cfg_type == 'pod':
        logger.warning("auto convert pod config -> UP config !!!")
        ConvertTool = POD2UP()
        yaml_data = ConvertTool.forward(yaml_data)
    else:
        pass
    # cfg check
    # return check_cfg(yaml_data)\
    # TODO check_cfg
    return yaml_data


class POD2UP:
    # convert the configs of POD to UP.
    def forward(self, pod_config):
        pod_c = pod_config
        del pod_c['version']
        del pod_c['dataset']['train']['dataset']['kwargs']['source']
        del pod_c['dataset']['test']['dataset']['kwargs']['source']
        # runtime setting
        pod_c['runtime'] = {}
        runner_phases = ['rank_init', 'random_seed', 'aligned', 'iter_base', 'device', 'async_norm', 'special_bn_init']
        if pod_c.get('fp16'):
            pod_c['runtime']['fp16'] = pod_c['fp16']
            del pod_c['fp16']
        for phase in runner_phases:
            if phase in pod_c:
                pod_c['runtime'][phase] = pod_c[phase]
                del pod_c[phase]
        # transformer setting
        trans_phases = ['mosaicv2']
        trans_type = {}
        for key in pod_c:
            if isinstance(pod_c[key], dict) and pod_c[key].get('type', None):
                trans_type[pod_c[key]['type']] = key
        for p in trans_phases:
            if p not in trans_type:
                continue
            if p == 'mosaicv2':
                pod_c[trans_type[p]]['type'] = 'mosaic'
        # ema setting
        self.changeVname(pod_c['ema'], 'ema_type', 'yolov5_ema', 'exp')
        self.changeVname(pod_c['ema'], 'ema_type', 'base', 'linear')
        self.cancelpar(pod_c['ema']['kwargs'], ['copy_init', 'use_double'])
        # for hook
        for i in range(len(pod_c['hooks'])):
            self.changeVname(pod_c['hooks'][i], 'type', 'yolox', 'yolox_noaug')
        # for dataset
        self.changeKname(pod_c['dataset']['dataloader']['kwargs'],
                         'with_work_init',
                         'worker_init')

        net_c = pod_c['net']
        num_level = 0
        for i, st in enumerate(net_c):
            if st['name'] == 'backbone':
                if 'yolox' not in st['type']:
                    net_c[i] = self.changeBK(st)
                    continue
                net_c[i] = self.changeDetModule(st)
                del net_c[i]['kwargs']['inplanes']
                del net_c[i]['kwargs']['ceil_mode']
            elif st['name'] == 'neck':
                num_level = st['kwargs'].get('num_level', 0)
                net_c[i] = self.changeDetModule(st)
            elif st['name'] == 'roi_head':
                net_c[i], new_post = self.changeRoI(st, num_level)
                net_c.insert(i + 1, new_post)
            else:
                net_c[i] = self.changeDetModule(st)
        pod_c['net'] = net_c
        # checkbyeys(pod_c)
        return pod_c

    def checkbyeys(self, config):
        print(json.dumps(config, indent=4))

    def changeDetModule(self, normal):
        tp = normal['type']
        if 'pod' in tp:
            normal['type'] = 'up.tasks.det.' + tp.split('pod.')[-1]
        return normal

    def changeBK(self, backb):
        tp = backb['type']
        backb['type'] = 'up.models.backbones.' + tp.split('.')[-1]
        return backb

    def changeRoI(self, roi_head, num_level=0):
        tp = 'yolox' if 'YoloX' in roi_head['type'] else roi_head['type']
        roi_head = self.changeDetModule(roi_head)
        # for network
        kwargs = copy.deepcopy(roi_head['kwargs'])
        kwargs_p = copy.deepcopy(roi_head['kwargs'])
        head_k = {}
        for key in kwargs.keys():
            if key == 'cfg':
                # init prior
                if 'init_prior' in kwargs[key]['cls_loss'].get('kwargs', {}):
                    init_prior = kwargs[key]['cls_loss']['kwargs']['init_prior']
                    head_k['init_prior'] = init_prior
                # activation type
                head_k['class_activation'] = kwargs[key]['cls_loss']['type'].split('_')[0]
                # fpn
                if num_level != 0:
                    head_k['num_level'] = num_level
                # anchor
                anchor_generator = kwargs[key].get('anchor_generator', {})
                anchor_ratios = anchor_generator.get('kwargs', {}).get('anchor_ratios', [])
                anchor_scales = anchor_generator.get('kwargs', {}).get('anchor_scales', [])
                num_anchors = len(anchor_ratios) * len(anchor_scales)
                if num_anchors > 0:
                    head_k.update({'num_anchors': num_anchors})
            elif key == 'dense_points':
                head_k.update({'num_point': kwargs[key]})
            else:
                head_k.update({key: kwargs[key]})
        roi_head['kwargs'] = head_k

        # for post
        post = {}
        post['prev'] = roi_head['name']
        post_type = {
            'YoloX': 'yolox_post',
            'RPN': 'rpn_post',
            'Retina': 'retina_post'
        }
        for key in post_type:
            if key in roi_head['type']:
                post['type'] = post_type[key]
        if tp == 'yolox':
            post['name'] = 'yolox_post'
            post = self.build_post_yolox(post, kwargs_p)
        else:
            post['name'] = 'post_process'
            post = self.build_post(post, kwargs_p)
        return roi_head, post

    def build_post(self, post, kwargs):
        kwargs = copy.deepcopy(kwargs)
        reserved_keys = ['num_classes', 'cfg']
        new_kwargs = {}
        for key in kwargs:
            if key in reserved_keys:
                new_kwargs[key] = kwargs[key]
        post['kwargs'] = new_kwargs
        return post

    def build_post_yolox(self, post, kwargs):
        kwargs = copy.deepcopy(kwargs)
        reserved_keys = ['num_classes', 'cfg']
        new_kwargs = {}
        for key in kwargs:
            if key in reserved_keys:
                new_kwargs[key] = kwargs[key]
        new_kwargs['cfg']['obj_loss'] = new_kwargs['cfg']['center_loss']
        del new_kwargs['cfg']['center_loss']
        new_kwargs['cfg']['anchor_generator'] = new_kwargs['cfg']['center_generator']
        del new_kwargs['cfg']['center_generator']
        new_kwargs['cfg']['roi_supervisor'] = new_kwargs['cfg']['fcos_supervisor']
        del new_kwargs['cfg']['fcos_supervisor']
        new_kwargs['cfg']['roi_predictor'] = new_kwargs['cfg']['fcos_predictor']
        del new_kwargs['cfg']['fcos_predictor']
        post['kwargs'] = new_kwargs
        return post

    def changeVname(self, conf, key, name1, name2):
        if conf[key] == name1:
            conf[key] = name2

    def changeKname(self, conf, name1, name2):
        if name1 in conf.keys():
            conf.update({name2: conf[name1]})
            del conf[name1]

    def cancelpar(self, conf, cancel_list):
        for cancel in cancel_list:
            if cancel in conf.keys():
                del conf[cancel]


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fin:
        a = yaml.load(fin, IncludeLoader)

    print(json.dumps(a, indent=4))
