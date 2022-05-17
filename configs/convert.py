from ruamel.yaml import YAML
import sys
from pathlib import Path
import os


def traversal(root_folder):
    file = Path(root_folder)
    if file.is_dir():
        for folder in os.listdir(root_folder):
            path = os.path.join(root_folder, folder)
            traversal(path)
    elif file.exists():
        if '.yaml' in root_folder:
            try:
                convert(root_folder)
            except Exception as r:
                print(root_folder, 'can not load')
                print(r)
    else:
        return


def convert(cfg):
    cfg_old = cfg
    yaml = YAML()
    with open(cfg_old, "r") as f:
        data = yaml.load(f)
    ##########################
    data = revise_ssl(data)
    ###########################
    cfg_new = cfg_old
    with open(cfg_new, "w") as f:
        yaml.dump(data, f)
    f.close()


def revise_cls(data):
    dataset = data['dataset']
    train_k = dataset['train']['dataset']['kwargs']
    test_k = dataset['test']['dataset']['kwargs']

    train_k['meta_file'] = 'images/meta/train.txt'
    test_k['meta_file'] = 'images/meta/val.txt'
    train_k['image_reader']['type'] = 'fs_pillow'
    test_k['image_reader']['type'] = 'fs_pillow'
    train_k['image_reader']['kwargs']['image_dir'] = 'images/train'
    test_k['image_reader']['kwargs']['image_dir'] = 'images/val'

    if 'memcached' in train_k['image_reader']['kwargs'].keys():
        train_k['image_reader']['kwargs']['memcached'] = False
    if 'memcached' in test_k['image_reader']['kwargs'].keys():
        test_k['image_reader']['kwargs']['memcached'] = False

    if 'saver' in data.keys():
        saver = data['saver']
        if 'pretrain_model' in saver.keys():
            saver['pretrain_model'] = 'pretrain_path'
    if 'mimic' in data.keys():
        mimic = data['mimic']
        if 'teacher' in mimic.keys():
            mimic['teacher']['teacher_weight'] = 'teacher.pth.tar'
    return data


def revise_ssl(data):
    dataset = data['dataset']
    train_k = dataset['train']['dataset']['kwargs']
    train_k['meta_file'] = 'images/meta/train.txt'
    train_k['image_reader']['type'] = 'fs_pillow'
    train_k['image_reader']['kwargs']['image_dir'] = 'images/train'
    if 'test' in dataset.keys():
        test_k = dataset['test']['dataset']['kwargs']
        test_k['meta_file'] = 'images/meta/val.txt'
        test_k['image_reader']['type'] = 'fs_pillow'
        test_k['image_reader']['kwargs']['image_dir'] = 'images/val'

    if 'saver' in data.keys():
        saver = data['saver']
        if 'pretrain_model' in saver.keys():
            saver['pretrain_model'] = 'pretrain_path'
    return data


def revise_det(data):
    dataset = data['dataset']
    train_k = dataset['train']['dataset']['kwargs']
    test_k = dataset['test']['dataset']['kwargs']

    if dataset['train']['dataset']['type'] == 'coco':
        train_k['meta_file'] = 'coco/annotations/instances_train2017.json'
        test_k['meta_file'] = 'coco/annotations/instances_val2017.json'
        train_k['image_reader']['kwargs']['image_dir'] = 'coco/train2017'
        test_k['image_reader']['kwargs']['image_dir'] = 'coco/val2017'
        if train_k['image_reader']['type'] == 'ceph_opencv':
            train_k['image_reader']['type'] = 'fs_pillow'
            test_k['image_reader']['type'] = 'fs_pillow'
        if 'cache' in train_k.keys():
            train_k['cache']['cache_dir'] = 'coco_cache'
            train_k['cache']['cache_name'] = 'coco2017_train.pkl'
        if 'cache' in test_k.keys():
            test_k['cache']['cache_dir'] = 'coco_cache'
            test_k['cache']['cache_name'] = 'coco2017_val.pkl'
        if 'evaluator'in test_k.keys():
            test_k['evaluator']['kwargs']['gt_file'] = test_k['meta_file']

    if 'saver' in data.keys():
        saver = data['saver']
        if 'pretrain_model' in saver.keys():
            saver['pretrain_model'] = 'pretrain_path'
        if 'resume_model' in saver.keys():
            saver['resume_model'] = 'resume_path'
    if 'mimic' in data.keys():
        mimic = data['mimic']
        if 'teacher' in mimic.keys():
            mimic['teacher']['teacher_weight'] = 'teacher.pth.tar'
    return data


def revise_det3d(data):
    ps_k = data['point_sampling']['kwargs']
    ps_k['root_path'] = 'kitti_path'
    ps_k['db_info_paths'] = 'kitti/kitti_infos/kitti_dbinfos_train.pkl'

    dataset = data['dataset']
    train_k = dataset['train']['dataset']['kwargs']
    test_k = dataset['test']['dataset']['kwargs']
    if dataset['train']['dataset']['type'] == 'kitti':
        train_k['meta_file'] = 'kitti/kitti_infos/kitti_infos_train.pkl'
        test_k['meta_file'] = 'kitti/kitti_infos/kitti_infos_val.pkl'
        train_k['image_reader']['kwargs']['image_dir'] = 'kitti/training'
        test_k['image_reader']['kwargs']['image_dir'] = 'kitti/training'
        if 'evaluator'in test_k.keys():
            test_k['evaluator']['kwargs']['gt_file'] = test_k['meta_file']

    if 'saver' in data.keys():
        saver = data['saver']
        if 'pretrain_model' in saver.keys():
            saver['pretrain_model'] = 'pretrain_path'
    return data


def revise_seg(data):
    dataset = data['dataset']
    train_k = dataset['train']['dataset']['kwargs']
    test_k = dataset['test']['dataset']['kwargs']
    if dataset['train']['dataset']['type'] == 'seg':
        train_k['meta_file'] = 'cityscapes/fine_train.txt'
        test_k['meta_file'] = 'cityscapes/fine_val.txt'
        train_k['image_reader']['kwargs']['image_dir'] = 'cityscapes'
        test_k['image_reader']['kwargs']['image_dir'] = 'cityscapes'
        train_k['seg_label_reader']['kwargs']['image_dir'] = 'cityscapes'
        test_k['seg_label_reader']['kwargs']['image_dir'] = 'cityscapes'
    if 'saver' in data.keys():
        saver = data['saver']
        if 'pretrain_model' in saver.keys():
            saver['pretrain_model'] = 'pretrain_path'
        if 'resume_model' in saver.keys():
            saver['resume_model'] = 'resume_path'
    return data


if __name__ == "__main__":
    path = sys.argv[1]
    traversal(path)
