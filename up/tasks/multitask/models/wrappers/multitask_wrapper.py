from up.utils.general.registry_factory import MODULE_WRAPPER_REGISTRY, MODULE_ZOO_REGISTRY
from up.utils.general.log_helper import default_logger as logger

from .wrapper_utils import build_model, instance_method


@MODULE_ZOO_REGISTRY.register('multitask_wrapper')
@MODULE_WRAPPER_REGISTRY.register('multitask_wrapper')
def multitask_wrapper(module, inplanes=None, cfg={}):
    idxs = cfg.get('idxs', [])
    debug = cfg.get('debug', False)
    add_prefix = cfg.get('add_prefix', '')

    if inplanes is not None:
        module['kwargs']['inplanes'] = inplanes

    def set_task_idx(self, idx):
        self.task_idx = idx

    def forward(self, inputs):
        t = self.__class__.__name__
        if len(self._support_task_idxs) == 0 or self.task_idx in self._support_task_idxs:
            if debug:
                logger.info(f'{t}: {self.task_idx} {self._support_task_idxs} forward')
            return self._multitask_forward(inputs)
        else:
            if debug:
                logger.info(f'{t}: {self.task_idx} skip')
            return {}

    if isinstance(module, dict):
        module = build_model(module['type'], module['kwargs'])

    assert not hasattr(module, 'set_task_idx'), f'{module.__class__.__name__} cannot have native set_task_idx'

    if debug:
        logger.info(f"{module.prefix}: {idxs}")

    module._support_task_idxs = idxs
    module.task_idx = 0

    module._multitask_forward = module.forward
    module.forward = instance_method(forward, module)
    module.set_task_idx = instance_method(set_task_idx, module)
    if add_prefix:
        module.prefix += add_prefix
    return module


@MODULE_WRAPPER_REGISTRY.register('key_replace')
def key_replace_wrapper(module, cfg={}):
    add_prefix = cfg.get('add_prefix', '')
    module.input_replace_dict = cfg.get('input', {})
    module.outpu_replace_dict = cfg.get('output', {})

    def forward(self, inputs):
        for k, v in self.input_replace_dict.items():
            inputs[v] = inputs.pop(k)
        outputs = self._forward(inputs)
        for k, v in self.outpu_replace_dict.items():
            outputs[v] = outputs[k]
        return outputs

    if isinstance(module, dict):
        module = build_model(module['type'], module['kwargs'])

    module._forward = module.forward
    module.forward = instance_method(forward, module)
    if add_prefix:
        module.prefix += add_prefix
    return module
