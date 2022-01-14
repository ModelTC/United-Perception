import yaml
import re
import os.path
import json
import copy


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


IncludeLoader.add_constructor('!include', IncludeLoader.include)


def load_yaml(path):
    with open(path, "r")as f:
        yaml_data = yaml.load(f, IncludeLoader)
    if 'version' in yaml_data.keys():
        return pod2up(yaml_data)
    else:
        return yaml_data
        # return yaml.load(f, IncludeLoader)
    # return pod2up(path)


def pod2up(pod_config):
    # f = open(pod_config,'r',encoding='utf-8')
    # pod_c = yaml.load(open(pod_config, 'r'), Loader=IncludeLoader)
    pod_c = pod_config
    del pod_c['version']
    # del pod_c['fp16']
    del pod_c['dataset']['train']['dataset']['kwargs']['source']
    del pod_c['dataset']['test']['dataset']['kwargs']['source']
    net_c = pod_c['net']
    for i, st in enumerate(net_c):
        if st['name'] == 'roi_head' or st['name'] == 'backbone':
            if st['name'] == 'backbone':
                net_c[i] = changeBK(st)
            elif st['name'] == 'roi_head':
                net_c[i], new_post = changeRoI(st)
                net_c.insert(i + 1, new_post)
        else:
            net_c[i] = changeNormal(st)
    pod_c['net'] = net_c
    # print(json.dumps(pod_c, indent=4))
    return pod_c


def changeNormal(normal):
    tp = normal['type']
    if 'pod' in tp:
        normal['type'] = 'eod.tasks.det.models.' + tp.split('pod.models.')[-1]
    return normal


def changeBK(normal):
    tp = normal['type']
    normal['type'] = 'eod.models.backbones.' + tp.split('.')[-1]
    return normal


def changeRoI(roi_head):
    tp = roi_head['type']
    roi_head['type'] = 'eod.tasks.det.models.' + tp.split('pod.models.')[-1]
    # for network
    kwargs = copy.deepcopy(roi_head['kwargs'])
    kwargs_p = copy.deepcopy(roi_head['kwargs'])
    head_k = {}
    head_k.update({'feat_planes': kwargs['feat_planes']})
    head_k.update({'num_classes': kwargs['num_classes']})
    head_k.update({'initializer': kwargs['initializer']})
    if 'init_prior' in kwargs['cfg']['cls_loss']['kwargs'].keys():
        init_prior = kwargs['cfg']['cls_loss']['kwargs']['init_prior']
        head_k.update({'init_prior': init_prior})
    num_anchor = len(kwargs['cfg']['anchor_generator']['kwargs']['anchor_ratios']) * \
        len(kwargs['cfg']['anchor_generator']['kwargs']['anchor_scales'])
    head_k.update({'num_anchors': num_anchor})
    head_k.update({'class_activation': kwargs['cfg']['cls_loss']['type'].split('_')[0]})
    roi_head['kwargs'] = head_k
    # for postprocess
    post = {}
    post.update({'name': 'post_process'})
    post.update({'prev': roi_head['name']})
    post.update({'type': kwargs_p['cfg']['roi_supervisor']['type'] + '_post'})
    del kwargs_p['feat_planes']
    del kwargs_p['initializer']
    post.update({'kwargs': kwargs_p})
    return roi_head, post


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fin:
        a = yaml.load(fin, IncludeLoader)

    print(json.dumps(a, indent=4))
