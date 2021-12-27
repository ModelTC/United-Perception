# Standard Library
import torch.nn as nn

from eod.utils.general.registry_factory import HOOK_REGISTRY
from eod.utils.general.hook_helper import Hook


@HOOK_REGISTRY.register('gradient_collector')
class GradientCollector(Hook):
    """ Visualize ground truthes, detection results
    """

    def __init__(self, runner, hook_head="roi_head", hook_cls_head="cls_subnet_pred"):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(GradientCollector, self).__init__(runner)

        def _backward_fn_hook(module, grad_in, grad_out):
            if hasattr(runner.model, 'module'):
                mm = runner.model.module
            else:
                mm = runner.model
            mm.post_process.cls_loss.collect_grad(grad_out[0])
        assert hasattr(runner.model, hook_head), f"hook_head: {hook_head} is not in current model"
        model_head = getattr(runner.model, hook_head)
        assert hasattr(model_head, hook_cls_head), f"hook_cls_head: {hook_cls_head} is not in current head {hook_head}"
        model_cls_head = getattr(model_head, hook_cls_head)
        assert isinstance(model_cls_head, nn.ModuleList) or isinstance(model_cls_head, nn.Module), 'hook_cls_head: {hook_cls_head} type must inherit from nn.Module' # noqa
        if isinstance(model_cls_head, nn.ModuleList):
            for sub_module in model_cls_head:
                sub_module.register_backward_hook(_backward_fn_hook)
        else:
            model_cls_head.register_backward_hook(_backward_fn_hook)
