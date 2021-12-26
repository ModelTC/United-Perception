import json
import numpy as np
import torch

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import ANCHOR_GENERATOR_REGISTRY
from eod.utils.general.global_flag import ALIGNED_FLAG


__all__ = [
    'AnchorGenerator',
    'BoxAnchorGenerator',
    'HandCraftAnchorGenerator',
    'ClusteredAnchorGenerator',
    'PointAnchorGenerator']


class AnchorGenerator(object):
    def __init__(self):
        self._num_anchors = None
        self._num_levels = None
        self._base_anchors = None

    def build_base_anchors(self, anchor_strides):
        """ build anchors over one cell
        """
        raise NotImplementedError

    def get_anchors(self, featmap_shapes, device=None):
        """
        Arguments:
          - featmap_shapes: (list of tuple of (h, w, h*w*a, stride))
        """
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    @property
    def base_anchors(self):
        return self._base_anchors

    @property
    def num_anchors(self):
        """The number of anchors per cell
        """
        return self._num_anchors

    @property
    def num_levels(self):
        return self._num_levels


class BoxAnchorGenerator(AnchorGenerator):

    def get_anchors(self, featmap_shapes, device=None):
        """
        Arguments:
          - featmap_shapes: (list of tuple of (h, w, h*w*a, stride))

        Returns:
          - mlvl_anchors: (list of anchors of (h*w*a, 4))
        """
        strides = [shp[-1] for shp in featmap_shapes]
        base_anchors = self.build_base_anchors(strides)
        # logger.info(f'{self._anchor_ratios}, {self._anchor_scales}, {self._anchor_strides}')
        mlvl_anchors = []
        for anchors_over_grid, featmap_shape, in zip(base_anchors, featmap_shapes):
            featmap_h = featmap_shape[0]
            featmap_w = featmap_shape[1]
            featmap_stride = featmap_shape[-1]
            anchors = self.get_anchors_over_plane(
                anchors_over_grid, featmap_h, featmap_w, featmap_stride, device=device)
            mlvl_anchors.append(anchors)
            # logger.info(f'featmap_shape:{featmap_shape}, anchor_shape:{anchors.shape}')
            # logger.debug('anchors:{}'.format(anchors.shape))
        return mlvl_anchors

    def get_anchors_over_plane(self,
                               anchors_over_grid,
                               featmap_h,
                               featmap_w,
                               featmap_stride,
                               dtype=torch.float32,
                               device=None):
        """
        Args:
        anchors_over_grid
        """
        # [A, 4], anchors over one pixel

        anchors_over_grid = torch.from_numpy(anchors_over_grid).to(device=device, dtype=dtype)
        # spread anchors over each grid
        shift_x = torch.arange(0, featmap_w * featmap_stride, step=featmap_stride, dtype=dtype, device=device)
        shift_y = torch.arange(0, featmap_h * featmap_stride, step=featmap_stride, dtype=dtype, device=device)
        # [featmap_h, featmap_w]
        shift_y, shift_x = torch.meshgrid((shift_y, shift_x))
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

        anchors_over_grid = anchors_over_grid.reshape(1, -1, 4)
        shifts = shifts.reshape(-1, 1, 4).to(anchors_over_grid)
        anchors_overplane = anchors_over_grid + shifts
        return anchors_overplane.reshape(-1, 4)

    def get_anchors_over_grid(self, ratios, scales, stride):
        """
        generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
        """
        stride = stride.item()
        if ALIGNED_FLAG.aligned:
            w = stride
            h = stride
            scales = torch.tensor(scales)
            x_center, y_center = 0, 0

            h_ratios = torch.sqrt(torch.tensor(ratios).to(torch.float64))
            w_ratios = 1 / h_ratios
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchors = [
                x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
                y_center + 0.5 * hs
            ]
            anchors = torch.stack(base_anchors, dim=-1).numpy()
        else:
            ratios = np.array(ratios)
            scales = np.array(scales)
            anchor = np.array([1, 1, stride, stride], dtype=np.float) - 1
            anchors = self._ratio_enum(anchor, ratios)
            anchors = np.vstack([self._scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
        return anchors

    def _ratio_enum(self, anchor, ratios):
        """enumerate a set of anchors for each aspect ratio wrt an anchor."""
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor, scales):
        """enumerate a set of anchors for each scale wrt an anchor."""
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _whctrs(self, anchor):
        """return width, height, x center, and y center for an anchor (window)."""
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """
        given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
        return anchors


@ANCHOR_GENERATOR_REGISTRY.register('hand_craft')
class HandCraftAnchorGenerator(BoxAnchorGenerator):
    def __init__(self, anchor_ratios, anchor_scales, anchor_strides=None):
        super(HandCraftAnchorGenerator, self).__init__()
        self._anchor_ratios = anchor_ratios
        self._anchor_scales = anchor_scales
        self._anchor_strides = anchor_strides
        self._num_anchors = len(anchor_ratios) * len(anchor_scales)
        if anchor_strides is not None:
            self.build_base_anchors(anchor_strides)

    def build_base_anchors(self, anchor_strides):
        self._anchor_strides = anchor_strides
        if getattr(self, '_base_anchors', None) is not None:
            return self._base_anchors
        self._num_levels = len(anchor_strides)
        self._base_anchors = []
        for idx, stride in enumerate(anchor_strides):
            anchors_over_grid = self.get_anchors_over_grid(self._anchor_ratios, self._anchor_scales, stride)
            self._base_anchors.append(anchors_over_grid)
        # logger.debug('base_anchors:{}'.format(self._base_anchors))
        return self._base_anchors

    def export(self):
        json_anchors = [anchors.tolist() for i, anchors in enumerate(self.base_anchors)]
        return {
            'anchor_type': 'hand_craft',
            'anchor_ratios': self._anchor_ratios,
            'anchor_scales': self._anchor_scales,
            'anchor_strides': self._anchor_strides,
            'anchors': json_anchors
        }


@ANCHOR_GENERATOR_REGISTRY.register('cluster')
class ClusteredAnchorGenerator(BoxAnchorGenerator):
    def __init__(self, num_anchors_per_level, num_levels, base_anchors_file):
        super(ClusteredAnchorGenerator, self).__init__()
        self._num_anchors = num_anchors_per_level
        self._num_levels = num_levels
        with open(base_anchors_file, 'r') as f:
            self._shapes = np.array(json.load(f)).reshape(num_levels, num_anchors_per_level, 2)
            assert self._shapes.size == num_anchors_per_level * num_levels * 2

    def load_anchors_over_grid(self, whs, stride):
        ws = whs[:, 0]
        hs = whs[:, 1]
        x_ctr = (stride - 1) / 2
        y_ctr = (stride - 1) / 2
        return self._mkanchors(ws, hs, x_ctr, y_ctr)

    def build_base_anchors(self, anchor_strides):
        if getattr(self, '_base_anchors', None) is not None:
            return self._base_anchors
        self._base_anchors = []
        for shape, stride in zip(self._shapes, anchor_strides):
            self._base_anchors.append(self.load_anchors_over_grid(shape, stride))

        return self._base_anchors

    def export(self):
        json_anchors = [anchor.tolist() for i, anchor in enumerate(self.base_anchors)]
        # return json_anchors
        return {
            'num_anchors': self.num_anchors,
            'num_levels': self.num_levels,
            'anchor_type': 'clustered',
            'anchors': json_anchors
        }


@ANCHOR_GENERATOR_REGISTRY.register('fcos')
class PointAnchorGenerator(AnchorGenerator):
    def __init__(self, dense_points=1, center=True):
        super(PointAnchorGenerator, self).__init__()
        self.dense_points = dense_points
        self.center = center

    def get_anchors(self, featmap_shapes, device=None):
        """
        Arguments:
          - featmap_shapes: (list of tuple of (h, w, h*w*a, stride))

        Returns:
          - locations: (list of Tensor of (h*w, 2))
        """
        locations = []
        for level, shape in enumerate(featmap_shapes):
            h, w, _, stride = shape
            locations_per_lever = self.compute_locations_per_lever(h, w, stride, device)
            locations.append(locations_per_lever)
        return locations

    def get_dense_locations(self, locations, stride, device):
        if self.dense_points <= 1:
            return locations
        center = 0
        step = stride // 4
        l_t = [center - step, center - step]
        r_t = [center + step, center - step]
        l_b = [center - step, center + step]
        r_b = [center + step, center + step]
        if self.dense_points == 4:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b], device=device)
        elif self.dense_points == 5:
            points = torch.cuda.FloatTensor([l_t, r_t, [center, center], l_b, r_b], device=device)
        else:
            logger.info("dense points only support 1, 4, 5")
        points.reshape(1, -1, 2)
        locations = locations.reshape(-1, 1, 2).to(points)
        dense_locations = points + locations
        dense_locations = dense_locations.view(-1, 2)
        return dense_locations

    def compute_locations_per_lever(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2 * self.center
        locations = self.get_dense_locations(locations, stride, device)
        return locations


def build_anchor_generator(cfg):
    return ANCHOR_GENERATOR_REGISTRY.build(cfg)
