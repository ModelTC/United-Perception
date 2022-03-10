import copy
import yaml

from up.utils.general.user_analysis_helper import get_task_from_cfg
from up.utils.general.tocaffe_helper import parse_resize_scale
from up.tasks.det.deploy import parser as up_parser
from up.utils.general.registry_factory import KS_PARSER_REGISTRY

__all__ = ['RavenParser']


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
