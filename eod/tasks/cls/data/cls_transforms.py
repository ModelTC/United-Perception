import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from eod.data.datasets.transforms import Augmentation
from eod.utils.general.registry_factory import AUGMENTATION_REGISTRY
import copy


__all__ = [
    'TorchAugmentation',
    'RandomResizedCrop',
    'RandomHorizontalFlip',
    'PILColorJitter',
    'TorchResize',
    'TorchCenterCrop',
    'TorchMixUp']


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


@AUGMENTATION_REGISTRY.register('torch_mixup')
class TorchMixUp(TorchAugmentation):
    def __init__(self, dataset, alpha=1.0, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes
        self.dataset = dataset
    def augment(self, data):
        output = copy.copy(data)
        idx_other = np.random.randint(0, len(self.dataset))
        data_other = self.dataset.get_input(idx_other)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        data['image'] = np.array(data['image'])
        data_other['image'] = np.array(data_other['image'])
        nh = max(data['image'].shape[0], data_other['image'].shape[0])
        nw = max(data['image'].shape[1], data_other['image'].shape[1])

        hmin = min(data['image'].shape[0], data_other['image'].shape[0])
        wmin = min(data['image'].shape[1], data_other['image'].shape[1])

        image_tmp = np.empty((nh, nw, 3), dtype=np.float32)
        image_tmp[:, :] = 0.

        image_tmp[0:data['image'].shape[0], 0:data['image'].shape[1]] = lam * data['image']
        image_tmp[0:data_other['image'].shape[0], 0:data_other['image'].shape[1]] += (1. - lam) * data_other['image']
        image_tmp = image_tmp.astype(np.uint8)
        image_tmp = Image.fromarray(image_tmp)

        label = torch.zeros(self.num_classes)
        label_other = torch.zeros(self.num_classes)
        label[data.gt] = 1
        label_other[data_other.gt] = 1
        output.gt = lam * label + (1 - lam) * label_other
        output.image = image_tmp
        return output


@AUGMENTATION_REGISTRY.register('torch_cutmix')
class TorchCutMix(TorchAugmentation):
    def __init__(self, dataset, alpha=1.0, num_classes=1000):
        self.alpha = alpha
        self.num_classes = num_classes
        self.dataset = dataset

    def rand_bbox(self, size, lam):
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def augment(self, data):
        output = copy.copy(data)
        idx_other = np.random.randint(0, len(self.dataset))
        data_other = self.dataset.get_input(idx_other)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        data['image'] = np.array(data['image'])
        data_other['image'] = np.array(data_other['image'])
        nh, nw = data['image'].shape[0], data['image'].shape[1]
        nh2, nw2 = data_other['image'].shape[0], data_other['image'].shape[1]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data['image'].shape, lam)
        bbx1 = min(nh2, bbx1)
        bbx2 = min(nh2, bbx2)
        bby1 = min(nw2, bby2)
        bby2 = min(nw2, bby2)

        data['image'][bbx1:bbx2, bby1:bby2] = data_other['image'][bbx1:bbx2, bby1:bby2] 
        data['image'] = data['image'].astype(np.uint8)
        data['image'] = Image.fromarray(data['image'])

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (nh * nw)

        label = torch.zeros(self.num_classes)
        label_other = torch.zeros(self.num_classes)
        label[data.gt] = 1
        label_other[data_other.gt] = 1
        output.gt = lam * label + (1 - lam) * label_other
        output.image = data.image
        return output


@AUGMENTATION_REGISTRY.register('torch_cutmix_mixup')
class TorchCutMixUp(TorchAugmentation):
    def __init__(self, dataset, mixup_alpha=1.0, cutmix_alpha=1.0, switch_prob=0.5, num_classes=1000):
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.dataset = dataset
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def augment(self, data):
        use_cutmix = np.random.rand() < self.switch_prob
        if use_cutmix:
            return self.cutmix(data)
        return self.mixup(data)
    
    def rand_bbox(self, size, lam):
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def mixup(self, data):
        self.alpha = self.mixup_alpha
        output = copy.copy(data)
        idx_other = np.random.randint(0, len(self.dataset))
        data_other = self.dataset.get_input(idx_other)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        data['image'] = np.array(data['image'])
        data_other['image'] = np.array(data_other['image'])
        nh = max(data['image'].shape[0], data_other['image'].shape[0])
        nw = max(data['image'].shape[1], data_other['image'].shape[1])

        hmin = min(data['image'].shape[0], data_other['image'].shape[0])
        wmin = min(data['image'].shape[1], data_other['image'].shape[1])

        image_tmp = np.empty((nh, nw, 3), dtype=np.float32)
        image_tmp[:, :] = 0.

        image_tmp[0:data['image'].shape[0], 0:data['image'].shape[1]] = lam * data['image']
        image_tmp[0:data_other['image'].shape[0], 0:data_other['image'].shape[1]] += (1. - lam) * data_other['image']
        image_tmp = image_tmp.astype(np.uint8)
        image_tmp = Image.fromarray(image_tmp)

        label = torch.zeros(self.num_classes)
        label_other = torch.zeros(self.num_classes)
        label[data.gt] = 1
        label_other[data_other.gt] = 1
        output.gt = lam * label + (1 - lam) * label_other
        output.image = image_tmp
        return output

    def cutmix(self, data):
        self.alpha = self.cutmix_alpha
        output = copy.copy(data)
        idx_other = np.random.randint(0, len(self.dataset))
        data_other = self.dataset.get_input(idx_other)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        data['image'] = np.array(data['image'])
        data_other['image'] = np.array(data_other['image'])
        nh, nw = data['image'].shape[0], data['image'].shape[1]
        nh2, nw2 = data_other['image'].shape[0], data_other['image'].shape[1]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data['image'].shape, lam)
        bbx1 = min(nh2, bbx1)
        bbx2 = min(nh2, bbx2)
        bby1 = min(nw2, bby2)
        bby2 = min(nw2, bby2)

        data['image'][bbx1:bbx2, bby1:bby2] = data_other['image'][bbx1:bbx2, bby1:bby2] 
        data['image'] = data['image'].astype(np.uint8)
        data['image'] = Image.fromarray(data['image'])

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (nh * nw)

        label = torch.zeros(self.num_classes)
        label_other = torch.zeros(self.num_classes)
        label[data.gt] = 1
        label_other[data_other.gt] = 1
        output.gt = lam * label + (1 - lam) * label_other
        output.image = data.image
        return output