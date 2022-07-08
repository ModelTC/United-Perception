import cv2
import copy

from up.data.data_utils import get_image_size
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY
from up.data.datasets.transforms import Augmentation


@AUGMENTATION_REGISTRY.register('letterbox')
class Letterbox(Augmentation):
    def __init__(self, max_size=640, scale_up=False, padding_val=114, stride=32):
        super(Augmentation, self).__init__()
        self.max_size = max_size
        self.scale_up = scale_up
        self.stride = stride
        self.color = (padding_val, padding_val, padding_val)

    def resize(self, boxes, r, dw, dh, w, h):
        boxes[:, :4] *= r
        boxes[:, 0] += dw
        boxes[:, 1] += dh
        boxes[:, 2] += dw
        boxes[:, 3] += dh
        w_mask = boxes[:, 2] - boxes[:, 0] > 1.
        h_mask = boxes[:, 3] - boxes[:, 1] > 1.
        boxes = boxes[w_mask & h_mask]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1e-3)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1e-3)
        return boxes

    def augment(self, data):
        output = copy.copy(data)
        img = data.image
        h0, w0 = img.shape[:2]

        r0 = self.max_size / max(h0, w0)

        if r0 != 1:
            img = cv2.resize(
                img,
                (int(w0 * r0), int(h0 * r0)),
                interpolation=cv2.INTER_AREA
                if r0 < 1 else cv2.INTER_LINEAR,
            )
        ori_shape = get_image_size(img)
        tgt_shape = data.tgt_shape.numpy()

        r = min(tgt_shape[0] / ori_shape[0], tgt_shape[1] / ori_shape[1])
        if not self.scale_up:
            r = min(r, 1.0)

        new_unpad = int(round(ori_shape[1] * r)), int(round(ori_shape[0] * r))
        dw, dh = (tgt_shape[1] - new_unpad[0]) / 2, (tgt_shape[0] - new_unpad[1]) / 2

        if r != 1:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border
        r = r * r0

        output.image = img
        output.scale_factor = r
        output.dw, output.dh = dw, dh

        return output
