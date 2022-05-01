import torch
import torch.nn as nn
from up.utils.env.dist_helper import env, broadcast, allgather
from up.utils.env.dist_helper import simple_group_split
from up.utils.general.registry_factory import (
    MODULE_WRAPPER_REGISTRY
)


@MODULE_WRAPPER_REGISTRY.register('moco')
class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, q_plane=2048, k_plane=2048, dim=128, K=65536,
                 m=0.999, T=0.07, mlp=False, group_size=8):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        group_size: size of the group to use ShuffleBN (default: 8, shuffle data across all gpus)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        dim_q = q_plane
        dim_k = k_plane
        if mlp:  # hack: brute-force replacement
            self.encoder_q_fc = nn.Sequential(nn.Linear(dim_q, dim_q), nn.ReLU(), nn.Linear(dim_q, dim))
            self.encoder_k_fc = nn.Sequential(nn.Linear(dim_k, dim_k), nn.ReLU(), nn.Linear(dim_k, dim))
        else:
            self.encoder_q_fc = nn.Sequential(nn.Linear(dim_q, dim))
            self.encoder_k_fc = nn.Sequential(nn.Linear(dim_k, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.encoder_q_fc.parameters(), self.encoder_k_fc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        rank = env.rank
        world_size = env.world_size
        self.group_size = world_size if group_size is None else min(
            world_size, group_size)

        assert world_size % self.group_size == 0
        self.group_idx = simple_group_split(
            world_size, rank, world_size // self.group_size)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.encoder_q_fc.parameters(), self.encoder_k_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys, self.group_size, self.group_idx)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        target_size = self.queue[:, ptr:ptr + batch_size].shape[1]
        self.queue[:, ptr:ptr + batch_size] = keys.t()[:, :target_size]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, self.group_size, self.group_idx)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        broadcast(idx_shuffle, 0, group_idx=self.group_idx)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = env.rank % self.group_size
        idx_this = idx_shuffle.reshape(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, self.group_size, self.group_idx)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = env.rank % self.group_size
        idx_this = idx_unshuffle.reshape(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, input):
        if isinstance(input, dict):
            input = input['image']
        im_q, im_k = input[:, 0], input[:, 1]
        im_q = im_q.contiguous()
        im_k = im_k.contiguous()
        # compute query features
        q = self.encoder_q({'image' : im_q})  # queries: NxC
        q = self.encoder_q_fc(q['features'][-1].mean(dim=[2, 3]))
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k({'image' : im_k})  # keys: NxC
            k = self.encoder_k_fc(k['features'][-1].mean(dim=[2, 3]))

            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return {'logits': logits}


@MODULE_WRAPPER_REGISTRY.register('moco_vit')
class MoCo_ViT(nn.Module):
    def __init__(self, encoder_q, encoder_k, dim=256, mlp_dim=4096, m=0.999, T=1.0, group_size=8):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo_ViT, self).__init__()

        self.T = T
        self.m = m

        self.base_encoder = encoder_q
        self.momentum_encoder = encoder_k

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        rank = env.rank
        world_size = env.world_size
        self.group_size = world_size if group_size is None else min(
            world_size, group_size)

        assert world_size % self.group_size == 0
        self.group_idx = simple_group_split(
            world_size, rank, world_size // self.group_size)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for i in range(num_layers):
            dim1 = input_dim if i == 0 else mlp_dim
            dim2 = output_dim if i == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if i < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)

    def forward(self, input):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        if isinstance(input, dict):
            input = input['image']
        x1, x2 = input[:, 0], input[:, 1]
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder()  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        # normalize
        q1 = nn.functional.normalize(q1, dim=1)
        k2 = nn.functional.normalize(k2, dim=1)
        # gather all targets
        k2 = concat_all_gather(k2, self.group_size, self.group_idx)
        # Einstein sum is more intuitive
        q1_k2_logits = torch.einsum('nc,mc->nm', [q1, k2]) / self.T

        # normalize
        q2 = nn.functional.normalize(q2, dim=1)
        k1 = nn.functional.normalize(k1, dim=1)
        # gather all targets
        k1 = concat_all_gather(k1, self.group_size, self.group_idx)
        # Einstein sum is more intuitive
        q2_k1_logits = torch.einsum('nc,mc->nm', [q2, k1]) / self.T

        logits = torch.stack([q1_k2_logits, q2_k1_logits])

        return {'logits': logits}


# utils
@torch.no_grad()
def concat_all_gather(tensor, group_size, group_idx):
    """gather the given tensor across the group"""

    tensors_gather = [torch.ones_like(tensor) for _ in range(group_size)]
    allgather(tensors_gather, tensor, group_idx)

    output = torch.cat(tensors_gather, dim=0)
    return output
