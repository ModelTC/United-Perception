import torch.nn as nn
import torch
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.env.dist_helper import env, barrier, broadcast, MASTER_RANK
from eod.utils.general.log_helper import default_logger as logger
import random


__all__ = ['Multiscale']


@MODULE_ZOO_REGISTRY.register('multi_scale')
class Multiscale(nn.Module):
    def __init__(self,
                 random_size=[10, 20],
                 scale_step=32,
                 scale_rate=10,
                 mode='bilinear',
                 align_corners=False):
        super(Multiscale, self).__init__()
        self.scale_step = scale_step
        self.scale_rate = scale_rate
        self.random_size = random_size
        self.count = 0
        self.mode = mode
        self.align_corners = align_corners
        self.cur_size = None
        if mode != 'bilinear':
            self.align_corners = None

    def get_target_size(self, input):
        _, _, h, w = input['image'].size()
        tensor = torch.LongTensor(2).cuda()
        if env.is_master():
            size = random.randint(*self.random_size)
            scale = float(size * self.scale_step) / max(h, w)
            size = [int(h * scale), int(w * scale)]
            tensor[0], tensor[1] = size[0], size[1]
        if env.world_size > 1:
            barrier()
            broadcast(tensor, MASTER_RANK)
        return [tensor[0].item(), tensor[1].item()]

    def rescale(self, input):
        b, _, h, w = input['image'].size()
        scale_h = self.cur_size[0] / float(h)
        scale_w = self.cur_size[1] / float(w)
        if scale_h == 1 and scale_w == 1:
            return
        input['image'] = nn.functional.interpolate(input['image'],
                                                   self.cur_size,
                                                   mode=self.mode,
                                                   align_corners=self.align_corners)
        for idx in range(b):
            if input.get('gt_bboxes', None) is not None:
                input['gt_bboxes'][idx][:, [0, 2]] *= scale_w
                input['gt_bboxes'][idx][:, [1, 3]] *= scale_h
            if input.get('gt_ignores', None) is not None:
                input['gt_ignores'][idx][:, [0, 2]] *= scale_w
                input['gt_ignores'][idx][:, [1, 3]] *= scale_h

    def forward(self, input):
        if self.training:
            self.count += 1
            if(self.count % self.scale_rate == 0) or self.cur_size is None:
                self.cur_size = self.get_target_size(input)
                logger.info(f"current input size {self.cur_size}")
            self.rescale(input)
        return({'image': input['image']})

    def get_outplanes(self):
        return([3])
