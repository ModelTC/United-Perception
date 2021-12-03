import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.model import accuracy as A
from eod.models.losses import build_loss

__all__ = ['BaseClsPostProcess']


@MODULE_ZOO_REGISTRY.register('base_cls_postprocess')
class BaseClsPostProcess(nn.Module):
    def __init__(self, cls_loss):
        super(BaseClsPostProcess, self).__init__()
        self.cls_loss = build_loss(cls_loss)
        self.prefix = self.__class__.__name__

    def get_acc(self, logits, targets):
        acc = A.accuracy(logits, targets)[0]
        return acc

    def get_loss(self, logits, targets):
        loss_info = {}
        loss = self.cls_loss(logits, targets)
        loss_info[f"{self.prefix}.loss"] = loss
        loss_info[f"{self.prefix}.accuracy"] = self.get_acc(logits, targets)
        return loss_info

    def get_test_output(self, logits):
        scores = F.softmax(logits, dim=1)
        # compute prediction
        _, preds = logits.data.topk(k=1, dim=1)
        preds = preds.view(-1)
        return {"preds": preds, "scores": scores}

    def forward(self, input):
        logits = input['logits']
        output = {}
        if self.training:
            targets = input['gt']
            return self.get_loss(logits, targets)
        else:
            results = self.get_test_output(logits)
        output.update(results)
        return output
