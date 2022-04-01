from up.utils.model.optimizer_helper import BaseOptimizer
from up.utils.general.registry_factory import OPTIMIZER_REGISTRY


__all__ = ['SegformerOptimizer']


@OPTIMIZER_REGISTRY.register('segformer')
class SegformerOptimizer(BaseOptimizer):
    def get_trainable_params(self, cfg_optim=None):
        special_param_group = cfg_optim.get('special_param_group', [])
        pconfig = cfg_optim.get('pconfig', None)
        if pconfig is not None:
            return self.param_group_all(self.model, pconfig)
        trainable_params = [{"params": []}] + [{"params": [], "lr": spg['lr'],
                                                "weight_decay": spg['weight_decay']} for spg in special_param_group]
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                gid = 0
                for i, spg in enumerate(special_param_group):
                    if spg['key'] in n:
                        gid = i + 1
                        break
                trainable_params[gid]["params"].append(p)
        return trainable_params
