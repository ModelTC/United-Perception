import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  # noqa E402
from matplotlib.lines import Line2D  # noqa E402
import warnings
from itertools import chain

from up.utils.general.registry_factory import VISUALIZER_REGISTRY

warnings.filterwarnings("ignore")

__all__ = ['OpenCVVisualizer', 'PLTVisualizer']


def vis_bad_case_helper(box, label, text, ax, color_list, box_alpha=1):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color_box = color_list[label % len(color_list), 0:3]
    ax.add_patch(
        plt.Rectangle((box[0], box[1]),
                      box[2] - box[0],
                      box[3] - box[1],
                      fill=True,
                      color=color_box,
                      linewidth=0.5,
                      alpha=box_alpha * 0.5))
    box_w = x2 - x1
    box_h = y2 - y1
    len_ratio = 0.2
    d = min(box_w, box_h) * len_ratio
    corners = list()
    # top left
    corners.append([(x1, y1 + d), (x1, y1), (x1 + d, y1)])
    # top right
    corners.append([(x2 - d, y1), (x2, y1), (x2, y1 + d)])
    # bottom left
    corners.append([(x1, y2 - d), (x1, y2), (x1 + d, y2)])
    # bottom right
    corners.append([(x2 - d, y2), (x2, y2), (x2, y2 - d)])
    line_w = 2 if d * 0.4 > 2 else d * 0.4
    for corner in corners:
        (line_xs, line_ys) = zip(*corner)
        ax.add_line(Line2D(line_xs, line_ys, linewidth=line_w, color=color_box))
        ax.text(x1, y1 - 5, text, fontsize=5,
                family='serif',
                bbox=dict(facecolor=color_box, alpha=0.4, pad=0,
                          edgecolor='none'),
                color='white')
    return ax


def vis_bad_case_with_text(img, instances, missing_list, predict_list, mix_root, class_names, thick=1, dpi=80):

    def get_axes():
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(img.shape[1] / dpi * 2, img.shape[0] / dpi * 2)
        axes_list = list(chain(*axes))
        title_list = [
            'ground_truth \n{class name}: {class id} ',
            'false negative/missing \n {class name}: {class id} ',
            'false positive/mistake \n {pred class name}: {pred class id}, {score} ',
            'true positive \n {pred class name}: {pred class id}, {score}'
        ]
        for ax, title in zip(axes_list, title_list):
            ax.axis('off')
            ax.set_title(title)
            ax.imshow(img)
        return fig, axes_list

    color_list = colormap(rgb=True) / 255
    fig, axes_list = get_axes()
    # visualize ground truth and false negatives
    for idx, ins in enumerate(instances):
        bbox = ins['bbox']
        label = ins['label']
        name = class_names[label]
        axes_list[0] = vis_bad_case_helper(bbox, label, f'{name}: {label}', axes_list[0], color_list)
        if idx in missing_list:
            axes_list[1] = vis_bad_case_helper(bbox, label, f'{name}: {label}', axes_list[1], color_list)
    # visualize false positives and true positives
    for ins in predict_list:
        bbox = ins['bbox']
        label = ins['label']
        name = class_names[label]
        score = round(ins['score'], 2)
        if not ins['is_right']:
            # false positive
            axes_list[2] = vis_bad_case_helper(bbox, label, f'{name}: {label}, {score}', axes_list[2], color_list)
        else:
            # true positives
            axes_list[3] = vis_bad_case_helper(bbox, label, f'{name}: {label}, {score}', axes_list[3], color_list)
    for ax in axes_list:
        ax.imshow(img)
    return fig


def colormap(rgb=False):
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301,
        0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500,
        0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
        0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000,
        0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
        0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000,
        0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000,
        1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
        0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
        0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429,
        0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


class BaseVisualizer(object):
    def __init__(self, class_names=None, vis_dir='vis_dir', thresh=0.9, show_box=True, show_class=True):
        self.class_names = class_names
        self.vis_dir = vis_dir
        self.thresh = thresh
        self.show_box = show_box
        self.show_class = show_class

    def get_class_string(self, class_index, score):
        if self.class_names:
            class_text = self.class_names[class_index]
        else:
            class_text = 'id:{:d}'.format(class_index)
        return class_text + ' {:0.2f}'.format(score).lstrip('0').rstrip('.0')

    def vis(self, im, boxes, classes, filename, ig_boxes=None):
        raise NotImplementedError


@VISUALIZER_REGISTRY.register('opencv')
class OpenCVVisualizer(BaseVisualizer):
    _GRAY = (218, 227, 218)
    _GREEN = (18, 127, 15)
    _WHITE = (255, 255, 255)

    def __init__(self,
                 class_names=None,
                 vis_dir='vis_dir',
                 thresh=0.5,
                 show_box=True,
                 show_class=True):
        super(OpenCVVisualizer, self).__init__(class_names, vis_dir, thresh, show_box, show_class)

    def vis_bbox(self, img, bbox, thick=1):
        """Visualizes a bounding box."""
        (x0, y0, w, h) = bbox
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        cv2.rectangle(img, (x0, y0), (x1, y1), self._GREEN, thickness=thick)
        return img

    def vis_class(self, img, pos, class_str, font_scale=0.35):
        """Visualizes the class."""
        x0, y0 = int(pos[0]), int(pos[1])
        # Compute text size.
        txt = class_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
        # Place text background.
        back_tl = x0, y0 - int(1.3 * txt_h)
        back_br = x0 + txt_w, y0
        cv2.rectangle(img, back_tl, back_br, self._GREEN, -1)
        # Show text.
        txt_tl = x0, y0 - int(0.3 * txt_h)
        cv2.putText(img, txt, txt_tl, font, font_scale, self._GRAY, lineType=cv2.LINE_AA)
        return img

    def vis(self, im, boxes, classes, filename, ig_boxes=None):
        """Constructs a numpy array with the detections visualized."""
        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.thresh:
            return im

        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.thresh:
                continue

            # show box (off by default)
            if self.show_box:
                im = self.vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

            # show class (off by default)
            if self.show_class:
                class_str = self.get_class_string(classes[i], score)
                im = self.vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        output_name = os.path.basename(filename)
        output_name = os.path.join(self.vis_dir, output_name)
        cv2.imwrite(output_name, im)


@VISUALIZER_REGISTRY.register('plt')
class PLTVisualizer(BaseVisualizer):
    def __init__(self,
                 class_names=None,
                 vis_dir='vis_dir',
                 thresh=0.5,
                 show_box=True,
                 show_class=True,
                 dpi=200,
                 box_alpha=0.0,
                 ext='pdf'):
        super(PLTVisualizer, self).__init__(class_names, vis_dir, thresh, show_box, show_class)
        self.dpi = dpi
        self.box_alpha = box_alpha
        self.ext = ext

    def vis(self, im, boxes, classes, filename, ig_boxes=None):
        """Visual debugging of detections."""
        if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.thresh):
            return

        color_list = colormap(rgb=True) / 255

        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / self.dpi, im.shape[0] / self.dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

        if boxes is None:
            sorted_inds = []  # avoid crash when 'boxes' is None
        else:
            # Display in largest to smallest order to reduce occlusion
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sorted_inds = np.argsort(-areas)

        if ig_boxes is not None:
            for bbox in ig_boxes:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1],
                                  fill=False,
                                  edgecolor='y',
                                  linewidth=0.5,
                                  alpha=self.box_alpha))

        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.thresh:
                continue

            # show box (off by default)
            color_box = color_list[classes[i] % len(color_list), 0:3]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=True,
                              edgecolor='None',
                              color=color_box,
                              linewidth=0.5,
                              alpha=self.box_alpha * 0.5))
            # draw corners
            x1, y1, x2, y2 = bbox[:4]
            len_ratio = 0.2
            box_w = x2 - x1
            box_h = y2 - y1
            d = min(box_w, box_h) * len_ratio
            corners = list()
            # top left
            corners.append([(x1, y1 + d), (x1, y1), (x1 + d, y1)])
            # top right
            corners.append([(x2 - d, y1), (x2, y1), (x2, y1 + d)])
            # bottom left
            corners.append([(x1, y2 - d), (x1, y2), (x1 + d, y2)])
            # bottom right
            corners.append([(x2 - d, y2), (x2, y2), (x2, y2 - d)])
            line_w = 2 if d * 0.4 > 2 else d * 0.4
            for corner in corners:
                (line_xs, line_ys) = zip(*corner)
                ax.add_line(Line2D(line_xs, line_ys, linewidth=line_w, color=color_box))

            if self.show_class:
                ax.text(bbox[0],
                        bbox[1] - 5,
                        self.get_class_string(classes[i], score),
                        fontsize=3,
                        family='serif',
                        bbox=dict(facecolor=color_box, alpha=0.4, pad=0, edgecolor='none'),
                        color='white')

        output_name = os.path.basename(filename) + '.' + self.ext
        output_name = os.path.join(self.vis_dir, '{}'.format(output_name))
        fig.savefig(output_name, dpi=self.dpi)
        plt.close('all')
