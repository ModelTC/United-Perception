try:
    from spring_aux.analytics.io import send_async as send
except Exception as err:
    print(err)

from up.utils.env.dist_helper import env
from up import __version__

UPINFO = {"name": "UP", "version": __version__, "network": [], "func": '', 'task': ''}


def get_network_from_cfg(cfg):
    network = []
    for net in cfg.get('net', []):
        mtype = net.get('type', '')
        if mtype.find('.') >= 0:
            cls_name = mtype.split('.')[-1]
            network.append(cls_name)
        else:
            network.append(mtype)
    return network


def get_subnetwork_from_cfg(cfg):
    subnet = {}
    for net in cfg.get('net', []):
        name = net.get('name', '')
        if name == '':
            continue
        if name == 'roi_head':
            subnet['num_anchors'] = net.get('kwargs', {}).get('num_anchors', 1)
            subnet['roi_feat_planes'] = net.get('kwargs', {}).get('feat_planes', 'none')
        if name == 'post_process' and net.get('prev', '') == 'roi_head':
            subnet['anchor_generator'] = net.get('kwargs', {}).get('cfg', {}).get('anchor_generator', {})
            subnet['roi_class_activation_type'] = 'sigmoid' if 'sigmoid' in net.get('kwargs', {}).get(
                'cfg', {}).get('cls_loss', {}).get('type', '') else 'softmax'
        if name == 'bbox_head':
            subnet['bbox_feat_planes'] = net.get('kwargs', {}).get('feat_planes', 'none')
            subnet['bbox_class_activation_type'] = 'sigmoid' if 'sigmoid' in net.get('kwargs', {}).get(
                'cfg', {}).get('cls_loss', {}).get('type', '') else 'softmax'
        # subnet[f'{name}_normalize'] = net.get('kwargs', {}).get('normalize', 'without normalize or with default.')
        mtype = net.get('type', '')
        if mtype.find('.') >= 0:
            net_name = mtype.split('.')[-1]
        else:
            net_name = mtype
        subnet[name] = net_name
    return subnet


def get_lossinfo_from_cfg(cfg):
    loss_info = {}
    for net in cfg.get('net', []):
        name = net.get('name', '')
        if name == '':
            continue
        kwargs = net.get('kwargs', {})
        for k in kwargs:
            if 'loss' in k:
                loss_type = kwargs[k].get('type', '')
                loss_info[name + f'_{k}'] = loss_type
        cfg = kwargs.get('cfg', {})
        for k in cfg:
            if 'loss' in k:
                loss_type = cfg[k].get('type', '')
                loss_info[name + f'_{k}'] = loss_type
    return loss_info


def get_supervisor_from_cfg(cfg):
    supervisor_info = {}
    for net in cfg.get('net', []):
        name = net.get('name', '')
        if name == '':
            continue
        kwargs = net.get('kwargs', {})
        for k in kwargs:
            if 'supervisor' in k:
                supervisor_type = kwargs[k].get('type', '')
                supervisor_info[name + f'_{k}'] = supervisor_type
        cfg = kwargs.get('cfg', {})
        for k in cfg:
            if 'supervisor' in k:
                supervisor_type = cfg[k].get('type', '')
                supervisor_info[name + f'_{k}'] = supervisor_type
    return supervisor_info


def get_datasetinfo_from_cfg(cfg):
    dataset_info = {}
    dataset_info['dataset_type'] = cfg['dataset'].get('train', {}).get('dataset', {}).get('type', '')
    return dataset_info


def get_trainerinfo_from_cfg(cfg):
    trainer_info = {}
    trainer_info['up_optimizer_type'] = cfg['trainer'].get('optimizer', {}).get('type', '')
    trainer_info['up_lr_scheduler'] = cfg['trainer'].get('lr_scheduler', {}).get('type', '')
    trainer_info['up_max_epoch'] = cfg['trainer'].get("max_epoch", 0)
    return trainer_info


def get_task_from_cfg(cfg):
    cfg['runtime'] = cfg.setdefault('runtime', {})
    task_info = ''
    if 'task_names' in cfg['runtime']:
        task_names = cfg['runtime']['task_names']
        if isinstance(task_names, list):
            for task_name in task_names:
                task_info = task_info + str(task_name) + '_'
            task_info = task_info[:-1]
        else:
            task_info = str(task_names)
    else:
        task_dict = {
            'coco': 'det',
            'custom': 'det',
            'lvis': 'det',
            'cls': 'cls',
            'seg': 'seg',
            'keypoint': 'kp',
            'kitti': '3d',
            'unknown': 'unknown'
        }
        dataset_type = cfg['dataset'].get('train', {}).get('dataset', {}).get('type', 'unknown')
        task_info = task_dict.get(dataset_type, 'unknown')
    return task_info


def get_predictor_from_cfg(cfg):
    predictor_info = {}
    for net in cfg.get('net', []):
        name = net.get('name', '')
        if name == '':
            continue
        kwargs = net.get('kwargs', {})
        for k in kwargs:
            if 'predictor' in k:
                predictor_type = kwargs[k].get('type', '')
                predictor_info[name + f'_{k}'] = predictor_type
        cfg = kwargs.get('cfg', {})
        for k in cfg:
            if 'predictor' in k:
                predictor_type = cfg[k].get('type', '')
                predictor_info[name + f'_{k}'] = predictor_type
        cfg_train = cfg.get('train', {})
        for k in cfg_train:
            if 'predictor' in k:
                predictor_type = cfg_train[k].get('type', '')
                predictor_info[name + f'_train_{k}'] = predictor_type
        cfg_test = cfg.get('test', {})
        for k in cfg_test:
            if 'predictor' in k:
                predictor_type = cfg_test[k].get('type', '')
                predictor_info[name + f'_test_{k}'] = predictor_type
    return predictor_info


def send_info(cfg=None, func=''):
    if cfg is not None:
        try:
            UPINFO['network'] = get_network_from_cfg(cfg)
            UPINFO['task'] = get_task_from_cfg(cfg)
            UPINFO['func'] = func
            UPINFO.update(get_subnetwork_from_cfg(cfg))
            UPINFO.update(get_lossinfo_from_cfg(cfg))
            UPINFO.update(get_datasetinfo_from_cfg(cfg))
            UPINFO.update(get_supervisor_from_cfg(cfg))
            UPINFO.update(get_predictor_from_cfg(cfg))
            UPINFO.update(get_trainerinfo_from_cfg(cfg))
        except:  # noqa
            pass
    if env.is_master():
        try:
            send(UPINFO)
        except:  # noqa
            pass
