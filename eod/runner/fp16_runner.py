import torch
from eod.utils.general.registry_factory import RUNNER_REGISTRY
from eod.utils.general.global_flag import FP16_FLAG

from .base_runner import BaseRunner


__all__ = ['FP16Runner']


@RUNNER_REGISTRY.register('fp16')
class FP16Runner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.fp16 = True
        FP16_FLAG.fp16 = True

    def forward(self, batch, return_output=False):
        with torch.cuda.amp.autocast(enabled=True):
            output = self.forward_model(batch)
        self._temporaries['last_output'] = output
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        loss = sum(losses)
        if return_output:
            return loss, output
        else:
            return loss

    def backward(self, loss):
        self.model.zero_grad()
        self.scaler.scale(loss).backward()
        self._hooks('after_backward', self.cur_iter, loss)
        return loss

    def update(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._hooks('after_update', self.cur_iter)

    def resume_fp16(self, ckpt):
        self.scaler.load_state_dict(ckpt['scaler'])

    def get_fp16_dump_dict(self):
        return {'scaler': self.scaler.state_dict()}
