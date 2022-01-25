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
import torch.nn as nn

# Import from pod
from eod.utils.general.cfg_helper import merge_opts_into_cfg
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.saver_helper import Saver
from eod.utils.general.tocaffe_utils import ToCaffe
from eod.utils.general.registry_factory import MODEL_WRAPPER_REGISTRY, TOCAFFE_REGISTRY, MODEL_HELPER_REGISTRY
from .user_analysis_helper import get_task_from_cfg


@MODEL_WRAPPER_REGISTRY.register('pod')
class Wrapper(torch.nn.Module):
    def __init__(self, detector):
        super(Wrapper, self).__init__()
        self.detector = detector

    def forward(self, image, return_meta=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        output = self.detector(input)
        print(f'detector output:{output.keys()}')
        base_anchors = output['base_anchors']
        blob_names = []
        blob_datas = []
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blobs') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
                print(f'blobs:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        if return_meta:
            return blob_names, base_anchors
        else:
            return blob_datas


@MODEL_WRAPPER_REGISTRY.register('cls')
class CLSWrapper(torch.nn.Module):
    def __init__(self, detector, add_softmax=False):
        super(CLSWrapper, self).__init__()
        self.detector = detector
        self.add_softmax = add_softmax
        if self.add_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        out = self.detector(input)
        if self.add_softmax:
            out = self.softmax(out['logits'])
        else:
            out = out['logits']
        return out


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
    for tf in transforms:
        if 'resize' in tf['type']:
            if task_type == 'det':
                scale = int(math.ceil(max(tf['kwargs']['scales']) / alignment) * alignment)
                max_size = int(math.ceil(tf['kwargs']['max_size'] / alignment) * alignment)
                return scale, max_size
            else:
                scale = int(math.ceil(tf['kwargs']['size'] / alignment) * alignment)
                return scale, scale
    else:
        raise ValueError('No resize found')


@TOCAFFE_REGISTRY.register('pod')
class PodToCaffe(object):
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
            self.model = MODEL_WRAPPER_REGISTRY['pod'](self.model)
            output_names, base_anchors = self.model(image, return_meta=True)
            self.output_names = output_names
            self.anchor_json = os.path.join(self.save_dir, 'anchors.json')
            with open(self.anchor_json, 'w') as f:
                json.dump(base_anchors, f, indent=2)
        elif self.task_type == 'cls':
            add_softmax = self.cfg.get('to_kestrel', {}).get('add_softmax', False)
            self.model = MODEL_WRAPPER_REGISTRY['cls'](self.model, add_softmax)
            self.model(image)

    def _convert(self):
        import spring.nart.tools.pytorch as pytorch
        input_names = ['data']
        if self.task_type == 'cls':
            self.output_names = ['out']
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


def to_caffe(config, save_prefix='model', input_size=None, input_channel=3):
    opts = config.get('args', {}).get('opts', [])
    config = merge_opts_into_cfg(opts, config)
    tocaffe_type = config.pop('tocaffe_type', 'pod')
    tocaffe_ins = TOCAFFE_REGISTRY[tocaffe_type](config,
                                                 save_prefix,
                                                 input_size,
                                                 None,
                                                 input_channel)
    caffemodel_name = tocaffe_ins.process()

    return caffemodel_name
