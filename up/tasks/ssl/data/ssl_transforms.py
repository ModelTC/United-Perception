import torch
import random
import copy
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps
from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY


__all__ = [
    'TwoCropsTransform',
    'MOCOv1',
    'MOCOv2',
    'MOCOv3',
    'SimCLRv1']


class TwoCropsTransform(Augmentation):
    def __init__(self):
        self.op = lambda x: x
        self.trans1 = None
        self.trans2 = None

    def augment(self, data):
        images = data['image']
        q = self.trans1(images)
        k = self.trans2(images)
        data['image'] = torch.stack([q, k])
        return data


@AUGMENTATION_REGISTRY.register('torch_mocov1')
class MOCOv1(TwoCropsTransform):
    def __init__(self, **kwargs):
        self.augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.trans1 = transforms.Compose(self.augmentation)
        self.trans2 = transforms.Compose(self.augmentation)


@AUGMENTATION_REGISTRY.register('torch_mocov2')
class MOCOv2(TwoCropsTransform):
    def __init__(self, **kwargs):
        self.augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.trans1 = transforms.Compose(self.augmentation)
        self.trans2 = transforms.Compose(self.augmentation)


@AUGMENTATION_REGISTRY.register('torch_mocov3')
class MOCOv3(TwoCropsTransform):
    def __init__(self, **kwargs):
        self.augmentation1 = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.augmentation2 = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.trans1 = transforms.Compose(self.augmentation1)
        self.trans2 = transforms.Compose(self.augmentation2)


@AUGMENTATION_REGISTRY.register('torch_simclrv1')
class SimCLRv1(TwoCropsTransform):
    def __init__(self, **kwargs):
        self.augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur([.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.trans1 = transforms.Compose(self.augmentation)
        self.trans2 = transforms.Compose(self.augmentation)


@AUGMENTATION_REGISTRY.register('torch_gaussian_blur')
class GaussianBlur():
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def augment(self, data):
        output = copy.copy(data)
        output.image = self.op(data.image)
        return output


@AUGMENTATION_REGISTRY.register('torch_solarize')
class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
