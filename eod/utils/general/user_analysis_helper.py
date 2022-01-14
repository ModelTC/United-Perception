import threading
from spring.nart.tools.io import send

from eod.utils.env.dist_helper import env
from eod import __version__

EODINFO = {"name": "EOD", "version": __version__, "network": [], "func": '', 'task': ''}


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
        subnet[f'{name}.normalize'] = net.get('kwargs', {}).get('normalize', 'without normalize or with default.')
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
    task_dict = {
        'coco': 'det',
        'custom': 'det',
        'lvis': 'det',
        'cls': 'cls',
        'seg': 'seg',
        'none': 'none'
    }
    dataset_type = cfg['dataset'].get('train', {}).get('dataset', {}).get('type', 'none')
    return task_dict[dataset_type]


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
            EODINFO['network'] = get_network_from_cfg(cfg)
            EODINFO['task'] = get_task_from_cfg(cfg)
            EODINFO['func'] = func
            EODINFO.update(get_subnetwork_from_cfg(cfg))
            EODINFO.update(get_lossinfo_from_cfg(cfg))
            EODINFO.update(get_datasetinfo_from_cfg(cfg))
            EODINFO.update(get_supervisor_from_cfg(cfg))
            EODINFO.update(get_predictor_from_cfg(cfg))
            EODINFO.update(get_trainerinfo_from_cfg(cfg))
        except:  # noqa
            pass
    if env.is_master():
        t = threading.Thread(target=send, args=(EODINFO, ))
        t.start()
