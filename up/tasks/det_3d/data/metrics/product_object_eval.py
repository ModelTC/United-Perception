from .kitti_object_eval_python import rotate_iou_gpu_eval
import numpy as np
import numba
import warnings

# from c3p.metrics import IOU_3D

warnings.filterwarnings('ignore')
label_dict = {
    'Car': 'Car',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
    'Bus': 'Bus',
    'Tricyclist': 'Tricyclist',
    'Truck': 'Truck'
}


def d3_box_overlap(boxes, qboxes, criterion=-1):
    boxes = boxes[:, [0, 2, 1, 3, 5, 4, 6]]
    qboxes = qboxes[:, [0, 2, 1, 3, 5, 4, 6]]
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def gt_to_bboxes_v1(gt_file):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    for obj in gt_file:
        cls = label_dict[obj['attribute']]
        bbox = [obj['center'][axis] for axis in ['x', 'y', 'z']] + [
            obj[dim] for dim in ['length', 'width', 'height', 'rotation']
        ]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


def gt_to_bboxes_v2(gt_file, index):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    objs = gt_file
    for i in range(objs['name'].shape[0]):
        cls = label_dict[objs['name'][i]]
        bbox = objs['location'][i].tolist() + objs['dimensions'][i].tolist() +\
            [objs['rotation_y'][i]]
        # if index>149:
        #     bbox[2]-=2
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


def gt_to_bboxes_drop_outofrange(gt_file, index, point_cloud_range=[0, -46.0, -1, 92, 46.0, 4]):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    objs = gt_file
    for i in range(objs['name'].shape[0]):
        if objs['location'][i][0] < point_cloud_range[0] or objs['location'][i][0] > point_cloud_range[3] \
                or objs['location'][i][1] < point_cloud_range[1] or objs['location'][i][1] > point_cloud_range[4]:
            continue
        cls = label_dict[objs['name'][i]]
        bbox = objs['location'][i].tolist() + objs['dimensions'][i].tolist() +\
            [objs['rotation_y'][i]]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


def pred_to_bboxes(pred_file):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    for i in range(pred_file['name'].shape[0]):
        cls = label_dict[pred_file['name'][i]]
        bbox = pred_file['location'][i].tolist() + pred_file['dimensions'][i].tolist() +\
            [pred_file['rotation_y'][i]] + [pred_file['score'][i]]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


cls_dict = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}


def pred_to_bboxes2(pred_file):
    bboxes_dict = {
        'Car': [],
        'Pedestrian': [],
        'Cyclist': [],
        'Tricyclist': [],
        'Truck': [],
        'Bus': []
    }
    for i, box in enumerate(pred_file):
        cls = cls_dict[box[8]]
        bbox = box[:8]
        bboxes_dict[cls].append(bbox)
    return bboxes_dict


class Eval_result(object):
    def __init__(self):
        self.tp = []
        self.fp = []
        self.fn = []
        self.ori = []
        self.ori_half = []
        self.precision = []
        self.recall = []
        self.aos = 0
        self.AP = 0
        self.aos_half = 0
        self.F1_score = 0
        self.IOU_precision = 0
        self.max_index = 0
        self.max_F1_score = 0
        self.IOU_score = []

    def update_results(self, rets):
        tp, fp, fn, ori, ori_half, IOU_score = rets
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.ori.append(ori)
        self.ori_half.append(ori_half)
        self.IOU_score.append(IOU_score)

    def compute_statistics(self, gt_boxes, pred_boxes, IOU_threshold):
        gt_boxes = np.array(gt_boxes)
        pred_boxes = np.array(pred_boxes)
        if gt_boxes.shape[0] == 0:
            tp = np.zeros([20])
            fp = np.zeros([20])
            if not pred_boxes.shape[0] == 0:
                for j, conf in enumerate(np.linspace(0, 0.95, 20)):
                    fp[j] = pred_boxes[pred_boxes[:, 7] > conf].shape[0]
            fn = np.zeros([20])
            ori = np.zeros([20])
            ori_half = np.zeros([20])
            IOU_score = np.zeros([20])
            return tp, fp, fn, ori, ori_half, IOU_score
        if pred_boxes.shape[0] == 0:
            tp = np.zeros([20])
            fp = np.zeros([20])
            fn = len(gt_boxes) * np.ones([20])
            ori = np.zeros([20])
            ori_half = np.zeros([20])
            IOU_score = np.zeros([20])
            return tp, fp, fn, ori, ori_half, IOU_score
        gt_boxes = np.array(gt_boxes)
        pred_boxes = np.array(pred_boxes)
        overlap = d3_box_overlap(gt_boxes, pred_boxes).astype(np.float64)
        # overlap= IOU_3D.compute_IOU_3d(gt_boxes, pred_boxes)
        # output:n*m matrix overlap_part[i,j]:3d_iou(gt_boxes[i],dt_boxes[j])
        tp, fp, fn, ori, ori_half, IOU_score = np.zeros([20]), np.zeros([
            20
        ]), np.zeros([20]), np.zeros([20]), np.zeros([20]), np.zeros([20])
        for j, conf in enumerate(np.linspace(0, 0.95, 20)):
            overlap_part = overlap[:, pred_boxes[:, 7] > conf]
            dt_boxes = pred_boxes[pred_boxes[:, 7] > conf]
            if overlap_part.shape[1] == 0:
                tp[j] = 0
                fp[j] = 0
                fn[j] = gt_boxes.shape[0]
                ori[j] = 0
                ori_half[j] = 0
                IOU_score[j] = 0
                continue
            matched_list = np.zeros([dt_boxes.shape[0]])
            for i, gt_box in enumerate(gt_boxes):
                cur_IOU = 1
                num_match = 0
                gt_is_matched = 0
                IOU_list = np.sort(overlap_part[i])
                while cur_IOU > IOU_threshold and num_match < dt_boxes.shape[0]:
                    cur_IOU = IOU_list[-1 - num_match]
                    max_IOU_index = overlap_part[i].tolist().index(cur_IOU)
                    if cur_IOU > IOU_threshold and matched_list[
                            max_IOU_index] == 0:
                        matched_list[max_IOU_index] = 1
                        tp[j] += 1
                        ori[j] += (1.0
                                   + np.cos(gt_boxes[i, 6]
                                            - dt_boxes[max_IOU_index, 6])) / 2.0
                        ori_half[j] += abs(np.cos(gt_boxes[i, 6] - dt_boxes[max_IOU_index, 6]))
                        IOU_score[j] += cur_IOU
                        gt_is_matched = 1
                        break
                    num_match += 1
                if not gt_is_matched:
                    fn[j] += 1
            fp[j] = matched_list.shape[0] - np.sum(matched_list)
        return tp, fp, fn, ori, ori_half, IOU_score

    def compute_pre_rec_map(self):
        fn = np.array(self.fn).sum(axis=0)
        fp = np.array(self.fp).sum(axis=0)
        tp = np.array(self.tp).sum(axis=0) + 1e-7  # avoid divide zero
        ori = np.array(self.ori).sum(axis=0)
        ori_half = np.array(self.ori_half).sum(axis=0)
        IOU_score = np.array(self.IOU_score)
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        ori_pre = ori / tp
        ori_pre_half = ori_half / tp
        cur_ori = ori_pre[0]
        cur_ori_half = ori_pre_half[0]
        cur_pre = self.precision[0]
        cur_rec = self.recall[0]
        self.AP = 0
        self.aos = 0
        self.aos_half = 0
        self.F1_score = np.zeros(self.precision.shape[0])
        for i in range(1, self.precision.shape[0]):
            self.AP += (cur_rec - self.recall[i]) * cur_pre
            self.aos += (cur_rec - self.recall[i]) * cur_ori
            self.aos_half += (cur_rec - self.recall[i]) * cur_ori_half
            cur_pre = self.precision[i]
            cur_ori = ori_pre[i]
            cur_rec = self.recall[i]
        self.AP += cur_rec * cur_pre
        self.aos += cur_rec * cur_ori
        self.aos_half += cur_rec * cur_ori_half
        # self.aos=np.sum(ori)/np.sum(tp)
        # self.aos_half=np.sum(ori_half)/np.sum(tp)
        self.IOU_precision = np.sum(IOU_score) / np.sum(tp)
        self.F1_score = 2 * (self.precision * self.recall) / (self.precision
                                                              + self.recall)
        self.max_F1_score = max(self.F1_score)
        self.max_index = self.F1_score.tolist().index(self.max_F1_score)


def evaluate_results(gt_file, pred_file, point_cloud_range, drop_out_range):
    IOU_thresholds = {
        'Car': 0.5,
        'Pedestrian': 0.3,
        'Cyclist': 0.3,
        'Tricyclist': 0.3,
        'Truck': 0.5,
        'Bus': 0.5
    }
    cls_dict = [
        'Car', 'Pedestrian', 'Cyclist', 'Tricyclist', 'Truck', 'Bus', 'Total'
    ]
    metrics_results = {}
    for cls in cls_dict:
        metrics_results[cls] = Eval_result()

    pred_dict = dict()
    for pred in pred_file:
        pred_dict[pred['velodyne_path']] = pred

    for i in range(len(gt_file)):

        print("current:%d total:%d ratio:%.3f" % (i, len(gt_file), i / len(gt_file)), flush=True)
        # gt_bboxes = gt_to_bboxes_v2(gt_file[i], i)
        # pred_bboxes = pred_to_bboxes(pred_file[i])
        gt_bboxes = {}
        if drop_out_range:
            gt_bboxes = gt_to_bboxes_drop_outofrange(gt_file[i][0], i, point_cloud_range)
        else:
            gt_bboxes = gt_to_bboxes_v2(gt_file[i][0], i)
        frame_id = gt_file[i][1]
        pred_bboxes = pred_to_bboxes(pred_dict[frame_id])
        # pred_bboxes = pred_to_bboxes2(pred_file)

        for k, v in IOU_thresholds.items():  # get tp fp fp ori for each class
            rets = metrics_results[k].compute_statistics(gt_bboxes[k],
                                                         pred_bboxes[k],
                                                         IOU_threshold=v)
            metrics_results[k].update_results(rets)
            metrics_results['Total'].update_results(rets)
    for cls in cls_dict:  # get AP Aos for each class
        if np.array(
                metrics_results[cls].tp).sum() == 0:  # remove the invaild cls
            del metrics_results[cls]
            continue
        metrics_results[cls].compute_pre_rec_map()

    return metrics_results
