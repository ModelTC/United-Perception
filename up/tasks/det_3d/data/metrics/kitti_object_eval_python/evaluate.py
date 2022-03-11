from up.tasks.det_3d.data.metrics.kitti_object_eval_python import kitti_common as kitti
from .eval import get_coco_eval_result, get_official_eval_result
from up.utils.general.petrel_helper import PetrelHelper


def _read_imageset_file(path):
    lines = []
    with PetrelHelper.open(path) as f:
        for line in f:
            lines.append(line)
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)
