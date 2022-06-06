import numpy as np
import copy


def get_scale_factor(scale, max_size, img_h, img_w):
    short = min(img_w, img_h)
    large = max(img_w, img_h)
    if short <= 0:
        scale_factor = 1.0
        return scale_factor
    if scale <= 0:
        scale_factor = 1.0
    else:
        scale_factor = min(scale / short, max_size / large)
    return scale_factor


def get_miss_rate_multi_size(tp, fp, scores, gts_list_i, class_i, matched_gt_minsize, watch_scale, fppi):
    """
    input: accumulated tps & fps
    len(tp) == len(fp) ==len(scores) == len(box)
    """
    image_num = gts_list_i['image_num']
    gt_num = gts_list_i['gt_num'][class_i]
    gt_class_i = gts_list_i[class_i]
    image_scale = gts_list_i['image_scale']
    N = len(fppi)
    maxfps = fppi * image_num
    mrs = np.zeros(N)
    fppi_scores = np.zeros(N)

    multi_size_metrics = {}
    gt_bboxes = []
    gt_scales = []
    for img_id in gt_class_i:
        gts = gt_class_i[img_id]['gts']
        gt_bboxes.extend([g['bbox'] for g in gts])
        gt_scales.extend([image_scale[img_id] for g in gts])
    gt_bboxes = np.stack(gt_bboxes)
    gt_scales = np.stack(gt_scales)
    box_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    box_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    gt_minsize = np.minimum(box_w, box_h)
    gt_minsize = gt_minsize * gt_scales
    assert gt_num == gt_minsize.shape[0], f'gt_num vs gt_minsize: {gt_num} vs {gt_minsize.shape}'
    gt_hist, edges = np.histogram(gt_minsize, bins=watch_scale)

    for i, f in enumerate(maxfps):
        idxs = np.where(fp > f)[0]
        if len(idxs) > 0:
            idx = idxs[0]  # the last fp@fppi
        else:
            idx = -1  # no fps, tp[-1]==gt_num
        mrs[i] = 1 - tp[idx] / gt_num
        fppi_scores[i] = scores[idx]

        this_fppi_matched_gt_minsize = copy.deepcopy(matched_gt_minsize)
        this_fppi_matched_gt_minsize = this_fppi_matched_gt_minsize[:idx]
        this_fppi_matched_gt_minsize = this_fppi_matched_gt_minsize[this_fppi_matched_gt_minsize > 0]
        matched_gt_hist, edges = np.histogram(this_fppi_matched_gt_minsize, bins=watch_scale)

        tmp_accumulate_recall = []
        tmp_accumulate_size = []
        for bin_i in range(gt_hist.shape[0]):
            percent = gt_hist[bin_i] * 1.0 / max(1.0, gt_num)
            recall = matched_gt_hist[bin_i] * 1.0 / (gt_hist[bin_i] + 0.000001)
            miss_rate = 1 - recall # noqa
            base_fp_head = f'fppi-{fppi[i]}-'
            base_fp_size_head = base_fp_head + 'size {:4d}-{:4d}-recall_percent {:.4f}'.format(
                watch_scale[bin_i], watch_scale[bin_i + 1], percent)
            multi_size_metrics[base_fp_size_head] = round(recall, 4)

            tmp_accumulate_recall.append([percent, recall])  # [0.001, 0.6]
            tmp_accumulate_size.append(watch_scale[bin_i])  # 0

        base_fp_head_accumulate_recall = []
        for i, size in enumerate(tmp_accumulate_size[:]):
            tmp_percent = [e[0] for e in tmp_accumulate_recall[i:]]
            tmp_recall = [e[1] for e in tmp_accumulate_recall[i:]]
            if i == 0:
                total_percent = sum(tmp_percent)  # total percent not always equal to 1.0 (because of round)
            tmp_recall_sum = round(
                sum([
                    recall * percent * total_percent / (sum(tmp_percent) + 1e-12)
                    for percent, recall in zip(tmp_percent, tmp_recall)
                ]), 4)

            base_fp_head_accumulate_recall.append(f'{size}-{watch_scale[-1]}: {tmp_recall_sum}')
        multi_size_metrics[base_fp_head + 'accumulate_recall'] = base_fp_head_accumulate_recall
    return multi_size_metrics
