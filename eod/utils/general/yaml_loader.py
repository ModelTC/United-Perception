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
    if get_task_from_cfg(cfg) == 'det':
        net = cfg.get('net', [])
        assert len(net) > 0, "net doesn't exist."
        net_parse, idx2name = {}, {}
        for idx in range(len(net)):
            name = net[idx]['name']
            net_parse[name] = net[idx]
            idx2name[name] = idx
        # NaiveRPN: num_classes
        roi_head_type = net_parse['roi_head']['type']
        if roi_head_type == 'NaiveRPN':
            logger.info("auto reset rpn num_classes = 2")
            cfg['net'][idx2name['roi_head']]['kwargs']['num_classes'] = 2
            cfg['net'][idx2name['post_process']]['kwargs']['num_classes'] = 2
        # roi_head: class_activation
        cls_loss_type = net_parse['post_process']['kwargs']['cfg']['cls_loss']['type']
        class_activation = 'sigmoid' if 'sigmoid' in cls_loss_type else 'softmax'
        logger.info('auto reset class activation')
        cfg['net'][idx2name['roi_head']]['kwargs']['class_activation'] = class_activation
        # roi_head: num_anchors
        anchors = cfg['net'][idx2name['post_process']]['kwargs']['cfg']['anchor_generator']
        anchor_ratios = anchors['kwargs']['anchor_ratios']
        anchor_scales = anchors['kwargs']['anchor_scales']
        num_anchors = len(anchor_ratios) * len(anchor_scales)
        logger.info('auto reset num anchor base on anchor generator')
        cfg['net'][idx2name['roi_head']]['kwargs']['num_anchors'] = num_anchors
    else:
        pass

    return cfg


IncludeLoader.add_constructor('!include', IncludeLoader.include)


def load_yaml(path):
    with open(path, "r")as f:
        yaml_data = yaml.load(f, IncludeLoader)
    if 'version' in yaml_data.keys():
        logger.warning("auto convert pod config -> UP config !!!")
        ConvertTool = POD2UP()
        yaml_data = ConvertTool.forward(yaml_data)
    else:
        pass
    # cfg check
    return check_cfg(yaml_data)


class POD2UP:
    # convert the configs of POD to UP.
    def forward(self, pod_config):
        pod_c = pod_config
        del pod_c['version']
        del pod_c['dataset']['train']['dataset']['kwargs']['source']
        del pod_c['dataset']['test']['dataset']['kwargs']['source']
        # runner setting
        pod_c['runtime'] = {}
        runner_phases = ['rank_init', 'random_seed', 'aligned', 'iter_base', 'device', 'async_norm', 'special_bn_init']
        if pod_c.get('fp16'):
            pod_c['runtime']['fp16'] = pod_c['fp16']
            del pod_c['fp16']
        for phase in runner_phases:
            if phase in pod_c:
                pod_c['runtime'][phase] = pod_c[phase]
                del pod_c[phase]

        net_c = pod_c['net']
        for i, st in enumerate(net_c):
            if st['name'] == 'roi_head' or st['name'] == 'backbone':
                if st['name'] == 'backbone':
                    net_c[i] = self.changeBK(st)
                elif st['name'] == 'roi_head':
                    net_c[i], new_post = self.changeRoI(st)
                    net_c.insert(i + 1, new_post)
            else:
                net_c[i] = self.changeNormal(st)
        pod_c['net'] = net_c
        # checkbyeys(pod_c)
        return pod_c

    def checkbyeys(self, config):
        print(json.dumps(config, indent=4))

    def changeNormal(self, normal):
        tp = normal['type']
        if 'pod' in tp:
            normal['type'] = 'eod.tasks.det.models.' + tp.split('pod.models.')[-1]
        return normal

    def changeBK(self, backb):
        tp = backb['type']
        backb['type'] = 'eod.models.backbones.' + tp.split('.')[-1]
        return backb

    def changeRoI(self, roi_head):
        tp = roi_head['type']
        roi_head['type'] = 'eod.tasks.det.models.' + tp.split('pod.models.')[-1]
        # for network
        kwargs = copy.deepcopy(roi_head['kwargs'])
        kwargs_p = copy.deepcopy(roi_head['kwargs'])
        head_k = {}
        for key in kwargs.keys():
            if key == 'cfg':
                continue
            elif key == 'dense_points':
                head_k.update({'num_point': kwargs[key]})
            else:
                head_k.update({key: kwargs[key]})
        if 'initializer' in kwargs.keys():
            if 'init_prior' in kwargs['cfg']['cls_loss']['kwargs'].keys():
                init_prior = kwargs['cfg']['cls_loss']['kwargs']['init_prior']
                head_k.update({'init_prior': init_prior})
            num_anchor = len(kwargs['cfg']['anchor_generator']['kwargs']['anchor_ratios']) * \
                len(kwargs['cfg']['anchor_generator']['kwargs']['anchor_scales'])
            head_k.update({'num_anchors': num_anchor})
            head_k.update({'class_activation': kwargs['cfg']['cls_loss']['type'].split('_')[0]})
            roi_head['kwargs'] = head_k
            # for retinanet and faster-rcnn
            post = {}
            post.update({'name': 'post_process'})
            post.update({'prev': roi_head['name']})
            post.update({'type': kwargs_p['cfg']['roi_supervisor']['type'] + '_post'})
            if 'feat_planes' in kwargs_p.keys():
                del kwargs_p['feat_planes']
            if 'initializer' in kwargs_p.keys():
                del kwargs_p['initializer']
            post.update({'kwargs': kwargs_p})
        """
        # wait for furture exploiting
        else:
            # for yolox
            post = {}
            post.update({'name': 'yolox_post'})
            post.update({'prev': roi_head['name']})
            post.update({'type': 'yolox_post'})
            if 'width' in kwargs_p.keys():
                del kwargs_p['width']
            if 'dense_points' in kwargs_p.keys():
                del kwargs_p['dense_points']
            if 'act_fn' in kwargs_p.keys():
                del kwargs_p['act_fn']
            cfg = kwargs_p['cfg']
            self.changeKname(cfg, 'center_loss', 'obj_loss')
            self.changeKname(cfg, 'center_generator', 'anchor_generator')
            self.changeKname(cfg, 'fcos_supervisor', 'roi_supervisor')
            self.changeKname(cfg, 'fcos_predictor', 'roi_predictor')
            kwargs_p['cfg'] = cfg
            post.update({'kwargs': kwargs_p})
        """
        return roi_head, post

    def changeKname(self, conf, name1, name2):
        if name1 in conf.keys():
            conf.updata({name2: conf[name1]})
            del conf[name1]


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fin:
        a = yaml.load(fin, IncludeLoader)

    print(json.dumps(a, indent=4))
