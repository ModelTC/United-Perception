# Standard Library
import warnings

# Import from third library
import torch
from torch.utils.checkpoint import checkpoint


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")
        return False
    return True


def fully_checkpoint_sequential(functions, segments, *inputs):
    r"""Modified version of torch.utils.checkpoint.checkpoint_sequential for memory efficiency.
    It is assumed that at least one of the inputs have requires_grad=True, so we can checkpoint
    all of the segments at ease.
    Please refer to https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential
    for more details.
    """
    assert check_backward_validity(inputs), "At least one of the inputs needs requires_grad=True"

    def run_function(start, end, functions):
        def forward(*inputs):
            input = inputs[0]
            for j in range(start, end + 1):
                input = functions[j](input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
