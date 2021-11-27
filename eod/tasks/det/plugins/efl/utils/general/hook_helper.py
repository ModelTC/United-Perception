# Standard Library

from eod.utils.general.registry_factory import HOOK_REGISTRY
from eod.utils.general.hook_helper import Hook
from eod.utils.general.log_helper import default_logger as logger


@HOOK_REGISTRY.register('gradient_collector')
class GradientCollector(Hook):
    """ Visualize ground truthes, detection results
    """

    def __init__(self, runner, hook_target=None):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(GradientCollector, self).__init__(runner)
        # ht = runner.model       # hook target
        # sub_tar = hook_target.split('.')
        # for tar in sub_tar:
        #     assert hasattr(ht, tar)
        #     ht = getattr(ht, tar)

        def _backward_fn_hook(module, grad_in, grad_out):
            if hasattr(runner.model, 'module'):
                runner.model.module.post_process.cls_loss.collect_grad(grad_out[0])
            else:
                runner.model.post_process.cls_loss.collect_grad(grad_out[0])
        runner.model.roi_head.cls_subnet_pred.register_backward_hook(_backward_fn_hook)
