import yaml
import numpy as np

from up.tasks.det.deploy import parser as up_parser
from up.utils.general.registry_factory import KS_PARSER_REGISTRY

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
