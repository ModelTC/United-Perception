from __future__ import division

# Standard Library
import copy

from up.utils.general.registry_factory import TOCAFFE_REGISTRY
from up.utils.env.dist_helper import env, barrier
from up.utils.env.gene_env import to_device
from up.utils.general.log_helper import default_logger as logger


def set_device_with_model(model, input):
    for weight in model.parameters():
        device = weight.device
        dtype = weight.dtype
        break
    if input['image'].device != device or input['image'].dtype != dtype:
        input = to_device(input, device=device, dtype=dtype)
    return input


def remove_bbox_head(model):

    def new_forward(model):

        def lambda_forward(input):
            input = copy.copy(input)
            # tocaffe requires no to_device operation
            input = set_device_with_model(model, input)
            for name, submodule in model.named_children():
                if 'bbox_head' in name:
                    logger.info('skip bbox head with name {}'.format(name))
                    return input
                output = submodule(input)
                input.update(output)
            return input

        return lambda_forward
    model.forward = new_forward(model)


def tocaffe(cfg, save_prefix, input_size, model=None, input_channel=3):
    remove_bbox_head(model)
    tocaffe_type = cfg.pop('tocaffe_type', 'base')
    tocaffe_ins = TOCAFFE_REGISTRY[tocaffe_type](cfg,
                                                 save_prefix,
                                                 input_size,
                                                 model,
                                                 input_channel)

    if env.is_master():
        tocaffe_ins.process()
    barrier()
    onnx_name = save_prefix + '.onnx'
    return onnx_name
