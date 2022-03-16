from __future__ import division

# Standard Library
import copy
import json
import math
import os
import time
import warnings

# Import from third library
import torch

# Import from pod
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.saver_helper import Saver
from up.utils.general.tocaffe_utils import ToCaffe
from up.utils.general.registry_factory import MODEL_WRAPPER_REGISTRY, TOCAFFE_REGISTRY, MODEL_HELPER_REGISTRY
from .user_analysis_helper import get_task_from_cfg

__all__ = ['ToCaffe']


def build_model(cfg):
    for module in cfg['net']:
        if 'heads' in module['type']:
            module['kwargs']['cfg']['tocaffe'] = True

    model_helper_ins = MODEL_HELPER_REGISTRY[cfg.get('model_helper_type', 'base')]
    model = model_helper_ins(cfg['net'])
    saver = Saver(cfg['saver'])
    state_dict = saver.load_pretrain_or_resume()
    model_dict = model.state_dict()
    loaded_num = model.load(state_dict['model'], strict=False)
    if loaded_num != len(model_dict.keys()):
        warnings.warn('checkpoint keys mismatch loaded keys:({len(model_dict.keys())} vs {loaded_num})')
    return model


def parse_resize_scale(dataset_cfg, task_type='det'):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.update(dataset_cfg.get('test', {}))
    alignment = dataset_cfg['dataloader']['kwargs'].get('alignment', 1)
    transforms = dataset_cfg['dataset']['kwargs']['transformer']
    input_h, input_w = 0, 0
    for tf in transforms:
        if 'keep_aspect_ratio' in tf['type']:
            if task_type == 'kp':
                input_h = int(math.ceil(tf['kwargs']['in_h'] / alignment) * alignment)
                input_w = int(math.ceil(tf['kwargs']['in_w'] / alignment) * alignment)
        if 'resize' in tf['type']:
            if task_type == 'det':
                scale = int(math.ceil(max(tf['kwargs']['scales']) / alignment) * alignment)
                max_size = int(math.ceil(tf['kwargs']['max_size'] / alignment) * alignment)
                input_h, input_w = scale, max_size
            elif task_type == 'seg':
                input_h = int(math.ceil(tf['kwargs']['size'][0] / alignment) * alignment)
                input_w = int(math.ceil(tf['kwargs']['size'][1] / alignment) * alignment)
            else:
                scale = int(math.ceil(tf['kwargs']['size'] / alignment) * alignment)
                input_h = input_w = scale
    if input_h == 0 and input_w == 0:
        raise ValueError('No resize found')
    return input_h, input_w


@TOCAFFE_REGISTRY.register('base')
class BaseToCaffe(object):
    def __init__(self, cfg, save_prefix, input_size, model=None, input_channel=3):
        self.cfg = copy.deepcopy(cfg)
        self.save_prefix = save_prefix
        self.input_size = input_size
        self.model = model
        self.input_channel = input_channel
        self.task_type = get_task_from_cfg(self.cfg)

    def prepare_input_size(self):
        if self.input_size is None:
            resize_scale = parse_resize_scale(self.cfg['dataset'], self.task_type)
            self.input_size = (self.input_channel, *resize_scale)

    def prepare_save_prefix(self):
        self.save_dir = 'tocaffe'
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_prefix = os.path.join(self.save_dir, self.save_prefix)
        logger.info(f'save_prefix:{self.save_prefix}')

    def _build_model(self):
        if self.model is None:
            self.model = build_model(self.cfg)
        for module in self.model.modules():
            module.tocaffe = True

    def run_model(self):
        # disable trace
        ToCaffe.prepare()
        time.sleep(10)
        self.model = self.model.eval().cpu().float()
        image = torch.randn(1, *self.input_size)
        if self.task_type == 'det':
            self.model = MODEL_WRAPPER_REGISTRY['det'](self.model)
            output_names, base_anchors = self.model(image, return_meta=True)
            # dump anchors only for detection
            self.anchor_json = os.path.join(self.save_dir, 'anchors.json')
            with open(self.anchor_json, 'w') as f:
                json.dump(base_anchors, f, indent=2)
        elif self.task_type == 'cls':
            add_softmax = self.cfg.get('to_kestrel', {}).get('add_softmax', False)
            self.model = MODEL_WRAPPER_REGISTRY['cls'](self.model, add_softmax)
            output_names, _ = self.model(image)
        elif self.task_type == 'seg':
            self.model = MODEL_WRAPPER_REGISTRY['seg'](self.model)
            output_names, _ = self.model(image, return_meta=True)
        elif self.task_type == 'kp':
            self.model = MODEL_WRAPPER_REGISTRY['kp'](self.model)
            output_names, _ = self.model(image)
        else:
            logger.error(f"{self.task_type} is not supported, [det, cls, kp, seg] \
               is supported, you must set runtime: task_names: cls or others")
            raise NotImplementedError
        self.output_names = output_names

    def _convert(self):
        import spring.nart.tools.pytorch as pytorch
        input_names = ['data']
        with pytorch.convert_mode():
            pytorch.convert(
                self.model, [self.input_size],
                filename=self.save_prefix,
                input_names=input_names,
                output_names=self.output_names,
                verbose=True
            )
        logger.info('=============tocaffe done=================')
        caffemodel_name = self.save_prefix + '.caffemodel'
        return caffemodel_name

    def process(self):
        self.prepare_input_size()
        self.prepare_save_prefix()
        self._build_model()
        self.run_model()
        return self._convert()
