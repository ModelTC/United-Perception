import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from eod.data.datasets.transforms import Augmentation
from eod.utils.general.registry_factory import AUGMENTATION_REGISTRY
import copy


__all__ = [
    'TorchAugmentation',
    'RandomResizedCrop',
    'RandomHorizontalFlip',
    'PILColorJitter',
    'TorchResize',
    'TorchCenterCrop']


class TorchAugmentation(Augmentation):
    def __init__(self):
        self.op = lambda x: x

    def augment(self, data):
        output = copy.copy(data)
        output.image = self.op(data.image)
        return output


@AUGMENTATION_REGISTRY.register('torch_random_resized_crop')
class RandomResizedCrop(TorchAugmentation):
    def __init__(self, size, **kwargs):
        self.op = transforms.RandomResizedCrop(size, **kwargs)


@AUGMENTATION_REGISTRY.register('torch_random_horizontal_flip')
class RandomHorizontalFlip(TorchAugmentation):
    def __init__(self, p=0.5):
        self.p = p

    def augment(self, data):
        output = copy.copy(data)
        if torch.rand(1) < self.p:
            output.image = TF.hflip(data.image)
        return output


@AUGMENTATION_REGISTRY.register('torch_color_jitter')
class PILColorJitter(TorchAugmentation):
    def __init__(self, brightness, contrast, saturation, hue):
        self.op = transforms.ColorJitter(brightness, contrast, saturation, hue)


@AUGMENTATION_REGISTRY.register('torch_resize')
class TorchResize(TorchAugmentation):
    def __init__(self, size):
        self.op = transforms.Resize(size)


@AUGMENTATION_REGISTRY.register('torch_center_crop')
class TorchCenterCrop(TorchAugmentation):
    def __init__(self, size, **kwargs):
        self.op = transforms.CenterCrop(size, **kwargs)
