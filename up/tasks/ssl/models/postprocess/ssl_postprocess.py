import torch.nn as nn
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss

__all__ = ['BaseSslPostProcess']


@MODULE_ZOO_REGISTRY.register('base_ssl_postprocess')
class BaseSslPostProcess(nn.Module):
    def __init__(self, ssl_loss, prefix=None):
        super(BaseSslPostProcess, self).__init__()
        if isinstance(ssl_loss, list):
            self.ssl_loss = nn.Sequential()
            for _loss in ssl_loss:
                self.ssl_loss.add_module(_loss['type'], build_loss(_loss))
        else:
            self.ssl_loss = build_loss(ssl_loss)
        self.prefix = prefix if prefix is not None else self.__class__.__name__

    def get_loss(self, input):
        loss_info = {}
        if isinstance(self.ssl_loss, nn.Sequential):
            loss = 0
            for name, s_loss in self.ssl_loss.named_parameters():
                loss = s_loss(input)
                loss_info[f"{self.prefix}_{name}.loss"] = loss
        else:
            loss = self.ssl_loss(input)
            loss_info[f"{self.prefix}.loss"] = loss
        return loss_info

    def forward(self, input):
        return self.get_loss(input)
