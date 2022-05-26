# Standard Library
import json
import os
# from re import I
import shutil

# Import from third library
import torch
from collections import OrderedDict

# Import from local
from .log_helper import default_logger as logger
from .registry_factory import SAVER_REGISTRY
from up.utils.general.petrel_helper import PetrelHelper


__all__ = ['Saver']


@SAVER_REGISTRY.register('base')
class Saver(object):
    task_type = 'det'

    def __init__(self, save_cfg, yml_path=None, work_dir='./'):
        # checkpoint dir
        self.save_cfg = self.prepend_work_dir(save_cfg, work_dir)
        self.work_dir = work_dir
        self.save_dir = save_cfg['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        if yml_path is not None and 's3://' not in yml_path:  # TODO, save cpeh data
            yml_name = os.path.basename(yml_path)
            dst_path = os.path.join(self.save_dir, yml_name)
            shutil.copy(yml_path, dst_path)

        Saver.task_type = self.save_cfg.get('task_type', 'det')
        self.auto_resume = self.save_cfg.get('auto_resume', False)
        self.running_config_file = os.path.join(self.save_dir, 'running_config.json')

    def prepend_work_dir(self, save_cfg, work_dir):

        def osp(path):
            return os.path.join(work_dir, path)

        save_cfg['save_dir'] = osp(save_cfg['save_dir'])
        save_cfg['results_dir'] = osp(save_cfg['results_dir'])

        return save_cfg

    @staticmethod
    def get_model_from_ckpt(ckpt_path):
        return Saver.load_checkpoint(ckpt_path)['model']

    def load_pretrain_or_resume(self):
        if self.auto_resume:
            last_checkpoint_path = self.find_last_checkpoint()
            if last_checkpoint_path is not None:
                logger.warning('Load checkpoint from {}'.format(last_checkpoint_path))
                return self.load_checkpoint(last_checkpoint_path)
            else:
                logger.warning('Not found any valid checkpoint yet')

        if 'resume_model' in self.save_cfg:
            logger.warning('Load checkpoint from {}'.format(self.save_cfg['resume_model']))
            state = self.load_checkpoint(self.save_cfg['resume_model'])
            return state
        elif 'pretrain_model' in self.save_cfg:
            state = self.load_checkpoint(self.save_cfg['pretrain_model'])
            logger.warning('Load checkpoint from {}'.format(self.save_cfg['pretrain_model']))
            output = {}
            if 'ema' in state:
                if "ema_state_dict" in state['ema']:
                    logger.info("Load ema pretrain model")
                    st = state['ema']['ema_state_dict']
                else:
                    st = state['model']
                output['ema'] = state['ema']
            else:
                st = state['model']
            output['model'] = st
            return output
        else:
            logger.warning('Load nothing! No weights provided {}')
            return {'model': {}}

    @staticmethod
    def load_checkpoint(ckpt_path):
        # assert os.path.exists(ckpt_path), f'No such file: {ckpt_path}'
        device = torch.cuda.current_device()
        # ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
        ckpt_dict = PetrelHelper.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
        return Saver.process_checkpoint(ckpt_dict)

    @staticmethod
    def process_checkpoint(ckpt_dict):
        """Load state_dict from checkpoint"""

        def prototype_convert(state_dict):
            is_convert = True
            for k in state_dict.keys():
                if 'classifier' in k:
                    is_convert = False
            if is_convert:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'fc' in k:
                        k = k.replace('fc', 'head.classifier')
                    else:
                        k = 'backbone.' + k
                    new_state_dict[k] = v
                return new_state_dict
            else:
                return state_dict

        def pod_resnet_convert(state_dict):
            convert_dict1 = {
                "conv1.weight": "layer0.0.weight",
                "bn1.weight": "layer0.1.weight",
                "bn1.bias": "layer0.1.bias",
                "bn1.running_mean": "layer0.1.running_mean",
                "bn1.running_var": "layer0.1.running_var",
                "bn1.num_batches_tracked": "layer0.1.num_batches_tracked",
                # stem
                "conv1.0.weight": "layer0.0.0.weight",
                "conv1.1.bias": "layer0.0.1.bias",
                "conv1.1.running_mean": "layer0.0.1.running_mean",
                "conv1.1.running_var": "layer0.0.1.running_var",
                "conv1.1.weight": "layer0.0.1.weight",
                "conv1.3.weight": "layer0.0.3.weight",
                "conv1.4.bias": "layer0.0.4.bias",
                "conv1.4.running_mean": "layer0.0.4.running_mean",
                "conv1.4.running_var": "layer0.0.4.running_var",
                "conv1.4.weight": "layer0.0.4.weight",
                "conv1.6.weight": "layer0.0.6.weight",
            }
            convert_dict2 = {
                "backbone.conv1.weight": "backbone.layer0.0.weight",
                "backbone.bn1.weight": "backbone.layer0.1.weight",
                "backbone.bn1.bias": "backbone.layer0.1.bias",
                "backbone.bn1.running_mean": "backbone.layer0.1.running_mean",
                "backbone.bn1.running_var": "backbone.layer0.1.running_var",
                "backbone.bn1.num_batches_tracked": "backbone.layer0.1.num_batches_tracked",
                # stem
                "backbone.conv1.0.weight": "backbone.layer0.0.0.weight",
                "backbone.conv1.1.bias": "backbone.layer0.0.1.bias",
                "backbone.conv1.1.running_mean": "backbone.layer0.0.1.running_mean",
                "backbone.conv1.1.running_var": "backbone.layer0.0.1.running_var",
                "backbone.conv1.1.weight": "backbone.layer0.0.1.weight",
                "backbone.conv1.3.weight": "backbone.layer0.0.3.weight",
                "backbone.conv1.4.bias": "backbone.layer0.0.4.bias",
                "backbone.conv1.4.running_mean": "backbone.layer0.0.4.running_mean",
                "backbone.conv1.4.running_var": "backbone.layer0.0.4.running_var",
                "backbone.conv1.4.weight": "backbone.layer0.0.4.weight",
                "backbone.conv1.6.weight": "backbone.layer0.0.6.weight",
            }
            is_convert = False
            count1 = 0
            count2 = 0
            for k in convert_dict1.keys():
                if k in state_dict:
                    count1 += 1
            for k in convert_dict2.keys():
                if k in state_dict:
                    count2 += 1

            if count1 >= 5 or count2 >= 5:  # num_batches_tracked maybe not exist
                is_convert = True
            if is_convert:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k in convert_dict1:
                        new_state_dict[convert_dict1[k]] = v
                    elif k in convert_dict2:
                        new_state_dict[convert_dict2[k]] = v
                    else:
                        new_state_dict[k] = v
                return new_state_dict
            else:
                return state_dict

        def remove_prefix(state_dict, prefix):
            """Old style model is stored with all names of parameters share common prefix 'module.'"""
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        if 'model' in ckpt_dict:
            state_dict = ckpt_dict['model']
        elif 'state_dict' in ckpt_dict:
            state_dict = ckpt_dict['state_dict']
        else:
            state_dict = ckpt_dict

        state_dict = remove_prefix(state_dict, 'module.')
        if Saver.task_type == 'cls':
            state_dict = prototype_convert(state_dict)
        state_dict = pod_resnet_convert(state_dict)
        ckpt_dict['model'] = state_dict

        return ckpt_dict

    def lns_latest_ckpt(self, ckpt_path, new_path):
        try:
            pwd = os.getcwd()
            absolute_ckpt_path = os.path.join(pwd, ckpt_path)
            absolute_new_path = os.path.join(pwd, new_path)
            if os.path.exists(absolute_new_path):
                os.system(f'rm {absolute_new_path}')
            os.system(f"ln -s {absolute_ckpt_path} {absolute_new_path}")
        except Exception as e:
            logger.warning(f'Failed to ln -s {ckpt_path} {new_path}')
            logger.warning(e)

    def rm_ckpt(self, ckpt_path):
        try:
            os.system(f'rm {ckpt_path}')
        except Exception as e:
            logger.warning(f'Failed to rm {ckpt_path}')
            logger.warning(e)

    def save(self, epoch, iter, **kwargs):
        """Save model checkpoint for one epoch"""
        os.makedirs(self.save_dir, exist_ok=True)
        # Assume we warmup for a epochs and training a+b epochs in total,
        # then our checkpoints are named of ckpt_e{-a+1}.pth ~ ckpt_e{b}.pth
        # if best in kwargs, we save the best ckpt as ckpt_best.path.auto
        if 'spacial_name' in kwargs:
            ckpt_path = os.path.join(self.save_dir, kwargs['spacial_name'] + '.pth')
        elif 'suffix' in kwargs:
            suffix = kwargs['suffix']
            ckpt_path = os.path.join(self.save_dir, 'ckpt_e{}-{}.pth'.format(epoch, suffix))
        elif 'auto_save' in kwargs:
            ckpt_path = os.path.join(self.save_dir, 'ckpt_{}.pth'.format(kwargs['auto_save']))
        else:
            ckpt_path = os.path.join(self.save_dir, 'ckpt_e{}.pth'.format(epoch))
        # since epoch not in kwargs
        kwargs['epoch'] = epoch
        kwargs['iter'] = iter
        kwargs['metric_val'] = kwargs.get('metric_val', -1)
        lns_latest_ckpt = kwargs.pop('lns', True)
        if os.path.exists(ckpt_path):
            self.rm_ckpt(ckpt_path)
        torch.save(kwargs, ckpt_path)
        if lns_latest_ckpt:
            latest_path = os.path.join(self.save_dir, 'ckpt_latest.pth')
            self.lns_latest_ckpt(ckpt_path, latest_path)
        return ckpt_path

    def save_model_arch(self, model):
        """Save model structure"""
        os.makedirs(self.save_dir, exist_ok=True)
        meta_path = os.path.join(self.save_dir, 'model_arch.txt')
        with open(meta_path, 'w') as fid:
            fid.write(str(model))

    def save_running_config(self, config):
        with open(self.running_config_file, 'w') as rcf:
            json.dump(config, rcf, indent=2)

    def find_last_checkpoint(self):
        last_ckpt_path = os.path.join(self.save_dir, "ckpt_latest.pth")
        if os.path.exists(last_ckpt_path):
            return last_ckpt_path
        else:
            return None


@SAVER_REGISTRY.register('ceph')
class CephSaver(Saver):
    def __init__(self, save_cfg, yml_path=None, work_dir='./'):
        super().__init__(save_cfg, yml_path, work_dir)

        self.ceph_dir = save_cfg['ceph_dir']

    def get_ceph_ckpt_dir(self, ckpt_path):
        return os.path.join(self.ceph_dir, os.path.abspath(ckpt_path)[1:])

    def save(self, epoch, iter, **kwargs):
        """Save model checkpoint for one epoch"""
        os.makedirs(self.save_dir, exist_ok=True)
        # Assume we warmup for a epochs and training a+b epochs in total,
        # then our checkpoints are named of ckpt_e{-a+1}.pth ~ ckpt_e{b}.pth
        # if best in kwargs, we save the best ckpt as ckpt_best.path.auto
        if 'suffix' in kwargs:
            suffix = kwargs['suffix']
            ckpt_path = os.path.join(self.save_dir, 'ckpt_e{}-{}.pth'.format(epoch, suffix))
        elif 'auto_save' in kwargs:
            ckpt_path = os.path.join(self.save_dir, 'ckpt_{}.pth'.format(kwargs['auto_save']))
        else:
            ckpt_path = os.path.join(self.save_dir, 'ckpt_e{}.pth'.format(epoch))
        # since epoch not in kwargs
        kwargs['epoch'] = epoch
        kwargs['iter'] = iter
        kwargs['metric_val'] = kwargs.get('metric_val', -1)
        lns_latest_ckpt = kwargs.pop('lns', True)

        # save
        # torch.save(kwargs, ckpt_path)
        ceph_ckpt_path = self.get_ceph_ckpt_dir(ckpt_path)
        try:
            PetrelHelper.save(kwargs, ckpt_path, ceph_ckpt_path)
        except Exception as e:  # noqa
            logger.warn(
                'Exception found during save checkpoint.\n'
                f'Exception info: {e}\n'
                'checkpoint will be save to local disk.'
            )
            ceph_ckpt_path = os.path.abspath(ckpt_path)
            PetrelHelper.save(kwargs, ckpt_path, ceph_ckpt_path)

        if lns_latest_ckpt:
            latest_path = os.path.join(self.save_dir, 'ckpt_latest.pth')
            self.lns_latest_ckpt(ckpt_path, latest_path)
        return ckpt_path

    def lns_latest_ckpt(self, ckpt_path, new_path):
        return super().lns_latest_ckpt(ckpt_path + '.ini', new_path + '.ini')

    def find_last_checkpoint(self):
        last_ckpt_path = os.path.join(self.save_dir, "ckpt_latest.pth")
        if os.path.exists(last_ckpt_path) or os.path.exists(last_ckpt_path + '.ini'):
            return last_ckpt_path
        else:
            return None
