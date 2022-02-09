# Standard library
import importlib
import types

# Import from pod
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(mtype, kwargs):
    # For traditional uses
    if mtype.find('.') >= 0:
        module_name, cls_name = mtype.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)
    else:  # For usage that models are registered by MODULE_ZOO_REGISTRY
        cfg = {'type': mtype, 'kwargs': kwargs}
        return MODULE_ZOO_REGISTRY.build(cfg)


def instance_method(func, obj):
    return types.MethodType(func, obj)


def recursive_set(model, key, module):
    def _recursive(model, key):
        if '.' not in key:
            model.add_module(key, module)
        else:
            cur, key = key.split('.', 1)
            _recursive(getattr(model, cur), key)

    _recursive(model, key)
