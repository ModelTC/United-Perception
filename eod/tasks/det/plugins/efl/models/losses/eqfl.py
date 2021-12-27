# Import from third library
import torch
import torch.nn.functional as F

# Import from pod
from eod.models.losses.loss import _reduce
from eod.tasks.det.models.losses.entropy_loss import GeneralizedCrossEntropyLoss
from eod.tasks.det.models.losses.focal_loss import dynamic_normalizer
from eod.utils.env.dist_helper import allreduce
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import LOSSES_REGISTRY


__all__ = ['EqualizedQualityFocalLoss']


@LOSSES_REGISTRY.register('equalized_quality_focal_loss')
class EqualizedQualityFocalLoss(GeneralizedCrossEntropyLoss):
    """
    Quality focal loss: https://arxiv.org/abs/2006.04388,
    """
    def __init__(self,
                 name='equalized_quality_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=1204,
                 focal_gamma=2.0,
                 scale_factor=8.0,
                 fpn_levels=5,
                 dynamic_normalizer=False):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

        # cfg for focal loss
        self.focal_gamma = focal_gamma

        # ignore bg class and ignore idx
        self.num_classes = num_classes - 1

        # cfg for efl loss
        self.scale_factor = scale_factor
        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels

        logger.info(f"build EqualizedQualityFocalLoss, focal_gamma: {focal_gamma}, scale_factor: {scale_factor}")

    def forward(self, input, target, reduction, normalizer=None, scores=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        assert reduction != 'none', 'Not Supported none reduction yet'
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c)
        self.target = target.reshape(-1)
        self.n_i, _ = self.input.size()
        scores = scores.reshape(-1)

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, 1:]

        expand_target = expand_label(self.input, self.target)
        sample_mask = (self.target != self.ignore_index)

        inputs = self.input[sample_mask]
        scores = scores[sample_mask]

        # cache for gradient collector
        self.cache_mask = sample_mask
        self.cache_target = expand_target

        # normlizer
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(inputs).to(inputs.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(inputs, self.target[sample_mask], 0.5, self.focal_gamma)

        pred = torch.sigmoid(inputs)
        map_val = 1 - self.pos_neg.detach()
        dy_gamma = self.focal_gamma + self.scale_factor * map_val
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)[sample_mask]
        # weighting factor
        wf = ff / self.focal_gamma

        # loss of negative samples
        loss = F.binary_cross_entropy_with_logits(inputs,
                                                  inputs.new_zeros(inputs.shape),
                                                  reduction='none') * pred.pow(ff.detach())
        # loss of positive samples
        pos_inds = torch.where(self.target[sample_mask] != 0)[0]
        pos_target = self.target[sample_mask][pos_inds].long() - 1
        quality_factor = scores[pos_inds] - pred[pos_inds, pos_target]
        loss[pos_inds, pos_target] = F.binary_cross_entropy_with_logits(inputs[pos_inds, pos_target],
                                                                        scores[pos_inds],
                                                                        reduction='none') * quality_factor.abs().pow(ff.detach()[pos_inds, pos_target]) # noqa
        loss = loss * wf.detach()
        return _reduce(loss, reduction=reduction, normalizer=normalizer)

    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        self.grad_buffer.append(grad_in.detach().permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes))
        if len(self.grad_buffer) == self.fpn_levels:
            target = self.cache_target[self.cache_mask]
            grad = torch.cat(self.grad_buffer[::-1], dim=1).reshape(-1, self.num_classes)

            grad = torch.abs(grad)[self.cache_mask]
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

            allreduce(pos_grad)
            allreduce(neg_grad)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)

            self.grad_buffer = []
