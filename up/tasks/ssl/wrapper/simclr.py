import torch
import torch.nn as nn
import torch.nn.functional as F
from up.utils.general.registry_factory import (
    MODULE_WRAPPER_REGISTRY
)


@MODULE_WRAPPER_REGISTRY.register('simclr')
class SimCLR(nn.Module):

    def __init__(self, encoder, mlp=True, num_mlp_layer=2, dim=2048, output_dim=128):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.mlp = mlp
        self.num_mlp_layer = num_mlp_layer

        if mlp:
            if num_mlp_layer == 1:
                self.encoder_fc = nn.Sequential(nn.Linear(dim, output_dim))
            elif num_mlp_layer > 1:
                layers = []
                for _ in range(num_mlp_layer - 1):
                    layers += [nn.Linear(dim, dim), nn.ReLU()]
                layers.append(nn.Linear(dim, output_dim))
                self.encoder_fc = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input, dict):
            input = input['image']
        input = torch.cat((input[:, 0], input[:, 1]), dim=0)
        features = self.encoder({'image' : input})['features'][-1].mean(dim=[2, 3])
        features = nn.functional.normalize(features, dim=1)
        if self.mlp:
            features = self.encoder_fc(features)

        features = features.view(features.shape[0], -1)

        labels = torch.cat([torch.arange(features.shape[0] // 2), torch.arange(features.shape[0] // 2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool)

        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / 0.07

        return {'logits': logits}
