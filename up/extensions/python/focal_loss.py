# Import from third library
import torch
from torch.autograd import Function

# Import from local
from ..ext import focal_loss


class SigmoidFocalLossFunction(Function):
    @staticmethod
    def forward(self, preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        """
        Arguments:
            preds: [batch * h * w * num_anchors, num_classes]
            targets: [batch * h * w * num_anchors]
            weight_pos: scalar tensor of normalizer
        """
        self.save_for_backward(preds, targets, weight_pos)
        weight_pos = weight_pos.cpu().item()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

        preds_size = preds.size()
        targets_size = targets.size()
        assert(preds_size[0] == targets_size[0]), f'{preds_size} vs {targets_size}'
        assert(preds_size[1] == self.num_classes), f'{preds_size} vs {self.num_classes}'

        losses = preds.new(preds_size[0], preds_size[1]).zero_()
        N = preds_size[0] * preds_size[1]

        assert(losses.is_contiguous())
        assert(preds.is_contiguous())
        assert(targets.is_contiguous())
        assert(preds.is_cuda and targets.is_cuda)

        focal_loss.sigmoid_forward_cuda(
            N,
            preds,
            targets,
            weight_pos,
            self.gamma,
            self.alpha,
            self.num_classes,
            losses)

        if reduction == 'none':
            return torch.cuda.FloatTensor(losses)
        else:
            return torch.cuda.FloatTensor([losses.sum()])

    @staticmethod
    def backward(self, grad_output):
        # grad_output: 1.0 / num_of_gpus
        preds, targets, weight_pos = self.saved_tensors
        preds_size = preds.size()
        grad_input = preds.new(preds_size[0], preds_size[1]).zero_()
        N = preds_size[0] * preds_size[1]

        assert(preds.is_contiguous())
        assert(targets.is_contiguous())
        assert(grad_input.is_contiguous())
        assert(preds.is_cuda and targets.is_cuda and grad_input.is_cuda)

        focal_loss.sigmoid_backward_cuda(
            N,
            preds,
            targets,
            grad_input,
            weight_pos.cpu().item(),
            self.gamma,
            self.alpha,
            self.num_classes)

        grad_input = grad_input * grad_output
        return grad_input, None, None, None, None, None, None


class SoftmaxFocalLossFunction(Function):

    @staticmethod
    def forward(self, preds, targets, weight_pos, gamma, alpha, num_classes):
        """
        Arguments:
            preds: [batch * h * w * num_anchors, num_classes]
            targets: [batch * h * w * num_anchors]
        """
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

        preds_size = preds.size()
        targets_size = targets.size()

        assert(preds_size[0] == targets_size[0])
        assert(preds_size[1] == self.num_classes)

        losses = preds.new(preds_size[0]).zero_()
        priors = preds.new(preds_size[0], preds_size[1]).zero_()

        N = preds_size[0] * preds_size[1]

        assert(losses.is_contiguous())
        assert(preds.is_contiguous())
        assert(targets.is_contiguous())
        assert(priors.is_contiguous())
        assert(preds.is_cuda and targets.is_cuda)

        focal_loss.softmax_forward_cuda(
            N,
            preds,
            targets,
            weight_pos.cpu().item(),
            self.gamma,
            self.alpha,
            self.num_classes,
            losses,
            priors)
        self.save_for_backward(preds, targets, weight_pos, priors)

        return torch.cuda.FloatTensor([losses.sum()])

    @staticmethod
    def backward(self, grad_output):
        # grad_output: 1.0 / num_of_gpus
        preds, targets, weight_pos, priors = self.saved_tensors
        preds_size = preds.size()
        grad_input = preds.new(preds_size[0], preds_size[1]).zero_()
        buff = preds.new(preds_size[0]).zero_()
        N = preds_size[0] * preds_size[1]

        assert(preds.is_contiguous())
        assert(targets.is_contiguous())
        assert(grad_input.is_contiguous())
        assert(buff.is_contiguous())
        assert(preds.is_cuda and targets.is_cuda and grad_input.is_cuda and buff.is_cuda)

        focal_loss.softmax_backward_cuda(
            N,
            preds,
            targets,
            grad_input,
            weight_pos.cpu().item(),
            self.gamma,
            self.alpha,
            self.num_classes,
            priors,
            buff)

        grad_input = grad_input * grad_output
        return grad_input, None, None, None, None, None
