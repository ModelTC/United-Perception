# Standard Library
import json
import os
import shutil

# Import from third library
import torch

# Import from local
from .log_helper import default_logger as logger
from .registry_factory import SAVER_REGISTRY


__all__ = ['Saver']


@SAVER_REGISTRY.register('base')
class Saver(object):
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
            else:
                st = state['model']
            output['model'] = st
            return output
        else:
            logger.warning('Load nothing! No weights provided {}')
            return {'model': {}}

    @staticmethod
    def load_checkpoint(ckpt_path):
        """Load state_dict from checkpoint"""

        def remove_prefix(state_dict, prefix):
            """Old style model is stored with all names of parameters share common prefix 'module.'"""
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        # assert os.path.exists(ckpt_path), f'No such file: {ckpt_path}'
        device = torch.cuda.current_device()
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))

        if 'model' in ckpt_dict:
            state_dict = ckpt_dict['model']
        elif 'state_dict' in ckpt_dict:
            state_dict = ckpt_dict['state_dict']
        else:
            state_dict = ckpt_dict

        state_dict = remove_prefix(state_dict, 'module.')
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
