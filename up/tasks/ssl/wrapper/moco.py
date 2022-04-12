import torch
import torch.nn as nn
import spring.linklink as link
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

        rank = link.get_rank()
        world_size = link.get_world_size()
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
        link.broadcast(idx_shuffle, 0, group_idx=self.group_idx)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = link.get_rank() % self.group_size
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
        gpu_idx = link.get_rank() % self.group_size
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


# utils
@torch.no_grad()
def concat_all_gather(tensor, group_size, group_idx):
    """gather the given tensor across the group"""

    tensors_gather = [torch.ones_like(tensor) for _ in range(group_size)]
    link.allgather(tensors_gather, tensor, group_idx)

    output = torch.cat(tensors_gather, dim=0)
    return output
