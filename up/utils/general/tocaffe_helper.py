from __future__ import division

# Standard Library
import copy
import json
import math
import os
import time
import warnings
import numpy as np

# Import from third library
import torch

# Import from pod
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.saver_helper import Saver
from up.utils.general.tocaffe_utils import ToCaffe
from up.utils.general.registry_factory import MODEL_WRAPPER_REGISTRY, TOCAFFE_REGISTRY, MODEL_HELPER_REGISTRY
from .user_analysis_helper import get_task_from_cfg

__all__ = ['BaseToCaffe', 'Point3DToCaffe']


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


def build_model_3d(cfg, input_dim=3):
    for module in cfg['net']:
        if 'heads' in module['type']:
            module['kwargs']['cfg']['tocaffe'] = True

    model_helper_ins = MODEL_HELPER_REGISTRY[cfg.get('model_helper_type', 'base')]
    vfe_model = model_helper_ins(cfg['net'][:1])
    backbone_model = model_helper_ins(cfg['net'][1:3])

    saver = Saver(cfg['saver'])
    state_dict = saver.load_pretrain_or_resume()

    vfe_model_dict = vfe_model.state_dict()
    backbone_model_dict = backbone_model.state_dict()
    if input_dim == 4:
        pfn_linear_weight = state_dict['model']['backbone3d.pfn_layers.0.linear.weight'].unsqueeze(-1).unsqueeze(-1)
        print('pfn_linear_weight:{}'.format(pfn_linear_weight.shape))
        state_dict['model']['backbone3d.pfn_layers.0.linear.weight'] = pfn_linear_weight
    vfe_loaded_num = vfe_model.load(state_dict['model'], strict=False)
    backbone_loaded_num = backbone_model.load(state_dict['model'], strict=False)

    if vfe_loaded_num != len(vfe_model_dict.keys()):
        warnings.warn('checkpoint keys mismatch loaded keys:({len(vfe_model_dict.keys())} vs {vfe_loaded_num})')
    if backbone_loaded_num != len(backbone_model_dict.keys()):
        warnings.warn(
            'checkpoint keys mismatch loaded keys:({len(backbone_model_dict.keys())} vs {backbone_loaded_num})')
    return vfe_model, backbone_model


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
                if type(tf['kwargs']['size']) is not list:
                    scale = int(math.ceil(tf['kwargs']['size'] / alignment) * alignment)
                    input_h = input_w = scale
                else:
                    input_h = int(math.ceil(tf['kwargs']['size'][0] / alignment) * alignment)
                    input_w = int(math.ceil(tf['kwargs']['size'][1] / alignment) * alignment)
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
            pytorch.convert.convert_v2(
                self.model, [self.input_size],
                filename=self.save_prefix,
                input_names=input_names,
                output_names=self.output_names,
                verbose=True,
                use_external_data_format=False
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


@TOCAFFE_REGISTRY.register('det_3d')
class Point3DToCaffe(object):
    def __init__(self, cfg, save_prefix, input_size, model=None, input_channel=3):
        self.cfg = copy.deepcopy(cfg)
        self.save_prefix = save_prefix
        self.task_type = get_task_from_cfg(self.cfg)
        self.num_point_features = self.cfg['num_point_features']
        voxel_cfg = self.cfg['to_voxel_train']['kwargs']
        self.max_points_per_voxel = voxel_cfg.get('max_points_per_voxel', 100)
        point_cloud_range = np.array(self.cfg['point_cloud_range'])
        voxel_size = voxel_cfg['voxel_size']
        voxel_size = np.array(voxel_size, dtype=np.float32)

        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.max_voxel_num = self.grid_size[0] * self.grid_size[1]
        # anchor setting
        anchor_cfg = self.cfg['net'][-1]['kwargs']['cfg'].get('anchor_generator', None)
        assert anchor_cfg is not None, "anchor generator config not exists."
        base_anchors_file = anchor_cfg['kwargs']['base_anchors_file']
        anchor_range = anchor_cfg['kwargs']['anchor_range']
        with open(base_anchors_file, 'r') as f:
            anchor_generator_cfg = np.array(json.load(f))
        if not anchor_generator_cfg[0].get(
                'anchor_offsets', None) or not anchor_generator_cfg[0].get('anchor_strides', None):
            anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_cfg]
            anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_cfg]
            anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_cfg]
            align_center = [config.get('align_center', False) for config in anchor_generator_cfg]
            anchor_offsets, anchor_strides = [], []
            for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                    [grid_size, grid_size, grid_size], anchor_sizes, anchor_rotations, anchor_heights, align_center):
                if align_center:
                    x_stride = (anchor_range[3] - anchor_range[0]) / grid_size[0]
                    y_stride = (anchor_range[4] - anchor_range[1]) / grid_size[1]
                    x_offset, y_offset = x_stride / 2, y_stride / 2
                else:
                    x_stride = (anchor_range[3] - anchor_range[0]) / (grid_size[0] - 1)
                    y_stride = (anchor_range[4] - anchor_range[1]) / (grid_size[1] - 1)
                    x_offset, y_offset = 0, 0
                anchor_offsets.append([x_offset, y_offset, 0])
                anchor_strides.append([x_stride, y_stride, 0])
            with open(base_anchors_file, 'w') as f:
                anchor_generator_cfg_pro = copy.deepcopy(anchor_generator_cfg)
                for idx in range(len(anchor_generator_cfg_pro)):
                    anchor_generator_cfg_pro[idx]["anchor_offsets"] = [anchor_offsets[idx]]
                    anchor_generator_cfg_pro[idx]["anchor_strides"] = [anchor_strides[idx]]
                json.dump(anchor_generator_cfg_pro.tolist(), f, indent=4)

        self.model = model
        self.input_channel = input_channel
        self.voxel_info = {
            'grid_size': self.grid_size,
            'num_point_feautures': self.num_point_features,
            'voxel_size': voxel_size,
            'point_cloud_range': point_cloud_range
        }

    def prepare_input_example(self):
        self.vfe_input_example = dict()
        self.vfe_input_example['voxel_infos'] = self.voxel_info
        self.vfe_input_example['voxels'] = torch.randn(
            self.max_voxel_num,
            self.max_points_per_voxel,
            self.num_point_features,
            1
        )
        self.vfe_input_example['voxel_coords'] = torch.randn(self.max_voxel_num, 4)
        self.vfe_input_example['voxel_num_points'] = torch.randn(self.max_voxel_num)
        # TODO: make sure the input size
        self.backbone2d_input_example = torch.randn(1, 64, int(self.grid_size[0]), int(self.grid_size[1]))

    def prepare_save_prefix(self):
        # vfe, 2d_backbone
        self.save_dir = 'tocaffe'
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_prefix = os.path.join(self.save_dir, self.save_prefix)
        logger.info(f'save_prefix:{self.save_prefix}')

    def _build_model(self):
        self.vfe_model, self.backbone_model = build_model_3d(self.cfg, input_dim=4)
        for model in [self.vfe_model, self.backbone_model]:
            for module in model.modules():
                module.tocaffe = True

    def run_model(self):
        # disable trace
        ToCaffe.prepare()
        time.sleep(10)
        self.vfe_model = self.vfe_model.eval().cpu().float()
        self.pfn_dims = self.vfe_model.backbone3d.num_point_features
        self.backbone_model = self.backbone_model.eval().cpu().float()
        if self.task_type == 'det_3d':
            self.vfe_model = MODEL_WRAPPER_REGISTRY['det_3d'](self.vfe_model, type='vfe')
            vfe_output_dict = self.vfe_model(self.vfe_input_example)
            self.pfn_model = vfe_output_dict['pfn']
            vfe_output_dict.pop('pfn')

            self.backbone_model = MODEL_WRAPPER_REGISTRY['det_3d'](self.backbone_model, type='backbone')
            backbone_output_dict = self.backbone_model(self.backbone2d_input_example)

        else:
            logger.error(f"{self.task_type} is not supported, [det, cls, kp, seg] \
               is supported, you must set runtime: task_names: cls or others")
            raise NotImplementedError
        self.vfe_output_names = vfe_output_dict.keys()
        self.backbone_output_names = backbone_output_dict.keys()

    def _convert(self):
        import spring.nart.tools.pytorch as pytorch
        pfn_input_name = "pfn_input"
        rpn_input_name = "rpn_input"
        with pytorch.convert_mode():
            a, b, c = map(int, (self.max_voxel_num, self.max_points_per_voxel, self.pfn_dims))
            print('vfe:', a, b, c)    # vfe: 33856 100 9
            pytorch.convert.export_onnx(
                self.pfn_model, [(b, c, 1)],
                self.save_prefix + ".pfn",
                input_names=[pfn_input_name],
                output_names=self.vfe_output_names,
                verbose=True,
                use_external_data_format=False
            )
            a, b, c, d = map(int, self.backbone2d_input_example.shape)
            print('backbone:', a, b, c, d)   # backbone: 1 64 184 184
            pytorch.convert.export_onnx(
                self.backbone_model, [(b, c, d)],
                self.save_prefix + ".rpn",
                input_names=[rpn_input_name],
                output_names=self.backbone_output_names,
                verbose=True,
                use_external_data_format=False
            )
        logger.info('=============tocaffe done=================')

        pfn_onnx_name = os.path.basename(self.save_prefix) + '.pfn.onnx'
        rpn_onnx_name = os.path.basename(self.save_prefix) + '.rpn.onnx'

        caffemodel_name = {
            'pfn': {
                'net_name': pfn_onnx_name,
                'input_name': pfn_input_name,
                'output_name': self.vfe_output_names,
            },
            'rpn': {
                'net_name': rpn_onnx_name,
                'input_name': rpn_input_name,
                'output_name': self.backbone_output_names
            }
        }
        return caffemodel_name

    def process(self):
        self.prepare_input_example()
        self.prepare_save_prefix()
        self._build_model()
        self.run_model()
        return self._convert()
