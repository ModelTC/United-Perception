import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model import accuracy as A
from up.models.losses import build_loss

__all__ = ['BaseClsPostProcess']


@MODULE_ZOO_REGISTRY.register('base_cls_postprocess')
class BaseClsPostProcess(nn.Module):
    def __init__(self, cls_loss, prefix=None):
        super(BaseClsPostProcess, self).__init__()
        if isinstance(cls_loss, list):
            self.cls_loss = nn.ModuleList()
            for _loss in cls_loss:
                self.cls_loss.append(build_loss(_loss))
        else:
            self.cls_loss = build_loss(cls_loss)
        self.prefix = prefix if prefix is not None else self.__class__.__name__

    def get_acc(self, logits, targets):
        acc = A.accuracy(logits, targets)[0]
        return acc

    def get_loss(self, logits, targets):
        loss_info = {}
        if isinstance(logits, list):
            loss = 0
            for idx, logit in enumerate(logits):
                if isinstance(self.cls_loss, nn.ModuleList):
                    assert len(logits) == len(self.cls_loss)
                    loss = self.cls_loss[idx](logit, targets[:, idx])
                else:
                    loss = self.cls_loss(logit, targets[:, idx])
                loss_info[f"{self.prefix}_head_{idx}.loss"] = loss
                loss_info[f"{self.prefix}_head_{idx}.accuracy"] = self.get_acc(logit, targets[:, idx])
        else:
            loss = self.cls_loss(logits, targets)
            loss_info[f"{self.prefix}.loss"] = loss
            loss_info[f"{self.prefix}.accuracy"] = self.get_acc(logits, targets)
        return loss_info

    def get_single_pred(self, logit):
        score = F.softmax(logit, dim=1)
        _, pred = logit.data.topk(k=1, dim=1)
        pred = pred.view(-1)
        return score, pred

    def get_test_output(self, logits):
        if isinstance(logits, list):
            scores = []
            preds = []
            for logit in logits:
                score, pred = self.get_single_pred(logit)
                preds.append(pred)
                scores.append(score)
        else:
            scores, preds = self.get_single_pred(logits)
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
