import torch
import torch.nn as nn
from up.utils.general.registry_factory import (
    MODULE_WRAPPER_REGISTRY
)


@MODULE_WRAPPER_REGISTRY.register('simsiam')
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, plane=2048, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.encoder = base_encoder
        # build a 3-layer projector
        prev_dim = plane
        self.encoder_fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True),  # hidden layer
                                        nn.Linear(pred_dim, dim))  # output layer

    def forward(self, input):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if isinstance(input, dict):
            input = input['image']
        x1, x2 = input[:, 0], input[:, 1]
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        # compute features for one view
        z1 = self.encoder({'image' : x1})  # NxC
        z1 = self.encoder_fc(z1['features'][-1].mean(dim=[2, 3]))
        z2 = self.encoder({'image' : x2})  # NxC
        z2 = self.encoder_fc(z2['features'][-1].mean(dim=[2, 3]))

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return {'p1': p1, 'p2': p2, 'z1': z1.detach(), 'z2': z2.detach()}