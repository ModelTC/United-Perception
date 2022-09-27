import torch
import torch.nn as nn

from up.utils.general.registry_factory import MODEL_WRAPPER_REGISTRY

__all__ = ['DetWrapper', 'ClsWrapper', 'SegWrapper', 'KpWrapper']


@MODEL_WRAPPER_REGISTRY.register('det')
class DetWrapper(torch.nn.Module):
    def __init__(self, detector):
        super(DetWrapper, self).__init__()
        self.detector = detector

    def forward(self, image, return_metas=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        output = self.detector(input)
        print(f'detector output:{output.keys()}')
        blob_names = []
        blob_datas = []
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blobs') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
                print(f'blobs:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        if return_metas:
            return blob_names
        else:
            return blob_datas


@MODEL_WRAPPER_REGISTRY.register('cls')
class ClsWrapper(torch.nn.Module):
    def __init__(self, detector, add_softmax=False):
        super(ClsWrapper, self).__init__()
        self.detector = detector
        self.add_softmax = add_softmax
        if self.add_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, image, return_metas=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        output = self.detector(input)
        print(f'detector output:{output.keys()}')
        if self.add_softmax:
            if isinstance(output['scores'], list):
                output['scores'] = [self.softmax(score) for score in output['scores']]
            else:
                output['scores'] = self.softmax(output['scores'])
        if isinstance(output['scores'], list):
            # blob_names = ['scores_%d' % i for i in range(len(output['scores']))]
            blob_names = ['scores']
            output['scores'] = torch.cat(output['scores'], dim=1)
        else:
            blob_names = ['scores']
        if return_metas:
            return blob_names
        else:
            return [output['scores']]


@MODEL_WRAPPER_REGISTRY.register('seg')
class SegWrapper(torch.nn.Module):
    def __init__(self, detector):
        super(SegWrapper, self).__init__()
        self.detector = detector

    def forward(self, image, return_metas=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        output = self.detector(input)
        print(f'model output:{output.keys()}')
        blob_names, blob_datas = [], []
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blob') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
                print(f'blob:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        if return_metas:
            return blob_names
        else:
            return blob_datas


@MODEL_WRAPPER_REGISTRY.register('kp')
class KpWrapper(torch.nn.Module):
    def __init__(self, detector):
        super(KpWrapper, self).__init__()
        self.detector = detector

    def forward(self, image, return_metas=False):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        output = self.detector(input)
        print(f'model output:{output.keys()}')
        blob_names, blob_datas = [], []
        output_names = sorted(output.keys())
        for name in output_names:
            if name.find('blob') >= 0:
                blob_names.append(name)
                blob_datas.append(output[name])
                print(f'blob:{name}')
        assert len(blob_datas) > 0, 'no valid output provided, please set "tocaffe: True" in your config'
        if return_metas:
            return blob_names
        else:
            return blob_datas
