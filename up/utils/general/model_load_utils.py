from up.utils.general.log_helper import default_logger as logger
from collections import OrderedDict
import copy


def load(model, other_state_dict, strict=False):
    """
    1. load resume model or pretained detection model
    2. load pretrained clssification model
    """
    other_state_dict.pop('model_cfg', None)
    model_keys = model.state_dict().keys()
    other_keys = other_state_dict.keys()
    shared_keys, unexpected_keys, missing_keys \
        = check_keys(model_keys, other_keys, 'model')

    # check shared_keys size
    new_state_dict = check_share_keys_size(model.state_dict(), other_state_dict, shared_keys, model_keys)
    # get share keys info from new_state_dict
    shared_keys, _, missing_keys \
        = check_keys(model_keys, new_state_dict.keys(), 'model')
    model.load_state_dict(new_state_dict, strict=strict)

    num_share_keys = len(shared_keys)
    if num_share_keys == 0:
        logger.info(
            'Failed to load the whole detection model directly, '
            'trying to load each part seperately...'
        )
        for mname, module in model.named_children():
            module_keys = module.state_dict().keys()
            other_keys = other_state_dict.keys()
            # check and display info module by module
            shared_keys, unexpected_keys, missing_keys, \
                = check_keys(module_keys, other_keys, mname)
            module_ns_dict = check_share_keys_size(
                module.state_dict(), other_state_dict, shared_keys, module_keys)
            shared_keys, _, missing_keys \
                = check_keys(module_keys, module_ns_dict.keys(), mname)

            module.load_state_dict(module_ns_dict, strict=strict)

            display_info(mname, shared_keys, unexpected_keys, missing_keys)
            num_share_keys += len(shared_keys)
    else:
        display_info("model", shared_keys, unexpected_keys, missing_keys)
    return num_share_keys


def check_keys(own_keys, other_keys, own_name):
    own_keys = set(own_keys)
    other_keys = set(other_keys)
    shared_keys = own_keys & other_keys
    unexpected_keys = other_keys - own_keys
    missing_keys = own_keys - other_keys
    return shared_keys, unexpected_keys, missing_keys


def check_share_keys_size(model_state_dict, other_state_dict, shared_keys, model_keys):
    new_state_dict = OrderedDict()
    for k in shared_keys:
        if k in model_keys:
            if model_state_dict[k].shape == other_state_dict[k].shape:
                new_state_dict[k] = other_state_dict[k]
    return new_state_dict


def display_info(mname, shared_keys, unexpected_keys, missing_keys):
    info = "Loading {}:{} shared keys, {} unexpected keys, {} missing keys.".format(
        mname, len(shared_keys), len(unexpected_keys), len(missing_keys))

    if len(missing_keys) > 0:
        info += "\nmissing keys are as follows:\n    {}".format("\n    ".join(missing_keys))
    logger.info(info)


def copy_module_weights(module, super_module):
    for (k1, v1), (k2, v2) in zip(module.state_dict().items(), super_module.state_dict().items()):
        assert v1.dim() == v2.dim()
        if v1.dim() == 0:
            v1.copy_(copy.deepcopy(v2.data))
        elif v1.dim() == 1:
            v1.copy_(copy.deepcopy(v2[:v1.size(0)].data))
        elif v1.dim() == 2:
            v1.copy_(copy.deepcopy(v2[:v1.size(0), :v1.size(1)].data))
        elif v1.dim() == 4:
            start, end = sub_filter_start_end(v2.size(3), v1.size(3))
            v1.copy_(copy.deepcopy(v2[:v1.size(0), :v1.size(1), start:end, start:end].data))


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end
