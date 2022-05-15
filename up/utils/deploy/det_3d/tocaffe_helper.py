from __future__ import division

# Standard Library
import copy
import json
import os
import time
import numpy as np
import warnings

# Import from third library
import torch

# Import from pod
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.saver_helper import Saver
from up.utils.general.tocaffe_utils import ToCaffe
from up.utils.general.registry_factory import MODEL_WRAPPER_REGISTRY, TOCAFFE_REGISTRY, MODEL_HELPER_REGISTRY

__all__ = ['BaseToCaffe']


def build_model(cfg, input_dim=3):
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


@MODEL_WRAPPER_REGISTRY.register('det_3d')
class Det3dWrapper(torch.nn.Module):
    def __init__(self, detector, type='backbone'):
        super(Det3dWrapper, self).__init__()
        self.detector = detector
        self.type = type

    def forward(self, example):
        # b, c, height, width = map(int, image.size())
        if self.type == 'backbone':
            input = {'spatial_features': example}
        elif self.type == 'vfe':
            input = example
        else:
            raise ValueError('invalid model type:{}'.format(type))
        print(f'before detector forward')
        output = self.detector(input)
        print(f'detector output:{output.keys()}')
        # base_anchors = output['base_anchors']
        blobs_dict = dict()
        blob_names = list()
        blob_datas = list()
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blobs') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
                blobs_dict[name.replace('blobs.', '')] = output[name]
                print(f'blobs:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        return blobs_dict


@TOCAFFE_REGISTRY.register('point')
class BaseToCaffe(object):
    def __init__(self, cfg, save_prefix, input_size, model=None, input_channel=3):
        self.cfg = copy.deepcopy(cfg)
        self.save_prefix = save_prefix
        self.task_type = 'det_3d'
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
        # anchor
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
                # anchor_generator_cfg_pro = json.load(f)
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
        batch_size = 1
        # voxel_features (voxels), voxel_coords (coords), voxel_num_points
        # torch.Size([6614, 100, 3]) torch.Size([6614, 4]) torch.Size([6614])
        self.pfn_dims = 8  # 9
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
        self.backbone2d_input_example = torch.randn(batch_size, 64, int(self.grid_size[0]), int(self.grid_size[1]))

    def prepare_save_prefix(self):
        # vfe, 2d_backbone
        self.save_dir = 'kestrel_model'
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_prefix = os.path.join(self.save_dir, self.save_prefix)
        logger.info(f'save_prefix:{self.save_prefix}')

    def _build_model(self):
        # if self.model is None:
        # build pfn and 2d backbone respectively
        self.vfe_model, self.backbone_model = build_model(self.cfg, input_dim=4)
        for model in [self.vfe_model, self.backbone_model]:
            for module in model.modules():
                module.tocaffe = True

    def run_model(self):
        # disable trace
        ToCaffe.prepare()
        time.sleep(10)
        self.vfe_model = self.vfe_model.eval().cpu().float()
        self.backbone_model = self.backbone_model.eval().cpu().float()
        # image = torch.randn(1, *self.input_size)
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


@TOCAFFE_REGISTRY.register('point_temp')
class TempBaseToCaffe(BaseToCaffe):
    def __init__(self, cfg, save_prefix, input_size, model=None, input_channel=3):
        super(TempBaseToCaffe, self).__init__(cfg, save_prefix, input_size, model, input_channel)

    def prepare_input_example(self):
        self.vfe_input_example = dict()
        # voxel_features (voxels), voxel_coords (coords), voxel_num_points
        # torch.Size([6614, 100, 3]) torch.Size([6614, 4]) torch.Size([6614])
        self.pfn_dims = 8  # 9
        self.vfe_input_example['voxel_infos'] = self.voxel_info
        self.vfe_input_example['voxels'] = torch.randn(
            self.max_voxel_num,
            self.max_points_per_voxel,
            self.num_point_features
        )
        self.vfe_input_example['voxel_coords'] = torch.randn(self.max_voxel_num, 4)
        self.vfe_input_example['voxel_num_points'] = torch.randn(self.max_voxel_num)
        # TODO: make sure the input size
        self.backbone2d_input_example = torch.randn(1, 64, int(self.grid_size[0]), int(self.grid_size[1]))

    def _build_model(self):
        # if self.model is None:
        # build pfn and 2d backbone respectively
        self.vfe_model, self.backbone_model = build_model(self.cfg)

        for model in [self.vfe_model, self.backbone_model]:
            for module in model.modules():
                # module.tocaffe = True
                module.tocaffe_temp = True

    def _convert(self):
        import spring.nart.tools.pytorch as pytorch
        with pytorch.convert_mode():
            a, b, c = map(int, (self.max_voxel_num, self.max_points_per_voxel, self.pfn_dims))
            print('vfe:', a, b, c)    # vfe: 33856 100 9
            pytorch.convert.export_onnx(
                self.pfn_model, [(b, c)],
                self.save_prefix + ".pfn",
                input_names=["pfn_input"],
                output_names=self.vfe_output_names,
                verbose=True,
                use_external_data_format=False
            )
            a, b, c, d = map(int, self.backbone2d_input_example.shape)
            print('backbone:', a, b, c, d)   # backbone: 1 64 184 184
            pytorch.convert.export_onnx(
                self.backbone_model, [(b, c, d)],
                self.save_prefix + ".rpn",
                input_names=["rpn_input"],
                output_names=self.backbone_output_names,
                verbose=True,
                use_external_data_format=False
            )
        logger.info('=============tocaffe done=================')

        caffemodel_name = self.save_prefix + '.caffemodel'
        return caffemodel_name
