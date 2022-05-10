import torch
import numpy as np
from collections.abc import Iterable, Mapping
from up.utils.general.log_helper import default_logger as logger


def detach(x):
    """detach from given tensor to block the trace route"""
    if torch.is_tensor(x):
        shape = tuple(map(int, x.shape))
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    elif isinstance(x, str) or isinstance(x, bytes):  # there is a dead loop when x is a str with len(x)=1n
        return x
    elif isinstance(x, np.ndarray):  # numpy recommends building array by calling np.array
        return np.array(list(map(detach, x)))
    elif isinstance(x, Mapping):
        return type(x)((k, detach(v)) for k, v in x.items())
    elif isinstance(x, Iterable):
        try:
            output = type(x)(map(detach, x))
        except Exception as e:
            logger.info(x)
            raise e
        return output

    else:
        return x


def get_model_hash(model_state_dict):
    try:
        from spring_analytics.model_lineage import get_model_meta
        model_hash = get_model_meta(model_state_dict)
    except Exception:
        return None
    return model_hash


def rewrite_onnx(onnx_prefix, model_hash):
    if model_hash is None:
        return
    try:
        import onnx
        onnx_file = onnx_prefix + '.onnx'
        onnx_model = onnx.load(onnx_file)
        onnx_model.doc_string += f'''
            model_hash: {model_hash}
        '''
        onnx.save(onnx_model, onnx_file)
    except Exception:
        return


class ToCaffe(object):
    _tocaffe = False

    @classmethod
    def disable_trace(self, func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if not self._tocaffe:
                return output
            else:
                return detach(output)
        return wrapper

    @classmethod
    def prepare(self):
        if self._tocaffe:
            return

        self._tocaffe = True

        # workaround to avoid onnx tracing tensor.shape
        torch.Tensor._shpae = torch.Tensor.shape

        @property
        def shape(self):
            return self.detach()._shpae
        torch.Tensor.shape = shape
