import torch.optim
import time
import os
import copy
import shutil
import json
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.hook_helper import build_hooks
from eod.utils.env.gene_env import set_random_seed, to_device
from eod.utils.env.dist_helper import barrier, all_gather, env, DistModule, reduce_gradients
from eod.data.metrics.base_evaluator import Metric
from eod.utils.general.cfg_helper import merge_opts_into_cfg, format_cfg
from eod.utils.env.gene_env import get_env_info
from eod.utils.general.fp16_helper import register_float_module
from eod.utils.general.registry_factory import (
    SAVER_REGISTRY,
    RUNNER_REGISTRY,
    EMA_REGISTRY,
    DATA_BUILDER_REGISTY,
    OPTIMIZER_REGISTRY,
    LR_SCHEDULER_REGISTY,
    MODEL_HELPER_REGISTRY
)
from eod.utils.general.global_flag import (
    ALIGNED_FLAG,
    ITER_BASE_FLAG,
    DIST_BACKEND,
    FP16_FLAG
)


__all__ = ['BaseRunner']


@RUNNER_REGISTRY.register('base')
class BaseRunner(object):
    def __init__(self, config, work_dir='./', training=True):
        self.args = config.get('args', {})
        self.backend = self.args.get('backend', 'dist')
        self.work_dir = work_dir
        self._progress = None
        self._eta = None
        self._tik = time.time()
        self.training = training
        self._temporaries = defaultdict(float)
        config = copy.deepcopy(config)
        self.config = self.upgrade_config(config)
        self.set_default_cfg()
        self.iter_base = self.config['runtime']['iter_base']
        self.device = self.config['runtime']['device']
        self.aligned = self.config['runtime']['aligned']
        self.fp16 = False
        self.build()

    def prepare_fp16(self):
        if self.config['runtime'].get('fp16', False):
            if self.backend == 'dist':
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            else:
                register_float_module(self.config['runtime']['fp16'].get('keep_batchnorm_fp32', True))
                pass
            self.fp16 = True
            FP16_FLAG.fp16 = True

    def save_running_cfg(self):
        self.saver.save_model_arch(self.model)
        self.saver.save_running_config(self.config)

    def set_default_cfg(self):
        self.config['runtime'] = self.config.setdefault('runtime', {})
        self.config['runtime']['rank_init'] = self.config['runtime'].setdefault('rank_init', True)
        self.config['runtime']['random_seed'] = self.config['runtime'].setdefault('random_seed', 131)
        self.config['runtime']['aligned'] = self.config['runtime'].setdefault('aligned', False)
        self.config['runtime']['iter_base'] = self.config['runtime'].setdefault('iter_base', False)
        self.config['runtime']['device'] = self.config['runtime'].setdefault('device', 'cuda')
        self.config['runtime']['async_norm'] = self.config['runtime'].setdefault('async_norm', False)
        self.config['runtime']['special_bn_init'] = self.config['runtime'].setdefault('special_bn_init', False)
        model_helper_cfg = self.config['runtime'].get('model_helper', {})
        model_helper_cfg['type'] = model_helper_cfg.get('type', 'base')
        model_helper_cfg['kwargs'] = model_helper_cfg.get('kwargs', {})
        self.config['runtime']['model_helper'] = model_helper_cfg

    def build(self):
        self.build_env()
        self.get_git_version()
        self.build_dataloaders()
        self.prepare_fp16()
        self.build_model()
        self.build_trainer()
        self.build_saver()
        self.build_hooks()
        self.build_ema()
        self.resume_model()
        self.prepare_dist_model()
        get_env_info()
        self.save_running_cfg()
        if not self.args.get('no_running_config', False):
            logger.info('Running with config:\n{}'.format(format_cfg(self.config)))

    def update_optimizer_cfg(self, cfg_optimizer):
        if not self.fp16:
            return cfg_optimizer
        if self.backend == 'dist':
            return cfg_optimizer
        elif self.backend == 'linklink':
            if cfg_optimizer['type'] == 'SGD':
                cfg_optimizer['type'] = 'FusedFP16SGD'
            if cfg_optimizer['type'] == 'AdamW':
                cfg_optimizer['type'] = 'FusedFP16AdamW'
            return cfg_optimizer
        else:
            raise NotImplementedError

    def build_optimizer(self, cfg_optimizer):
        register_type = cfg_optimizer.pop('register_type', 'base')
        cfg_optimizer = self.update_optimizer_cfg(cfg_optimizer)
        optimizer_ins = OPTIMIZER_REGISTRY[register_type](cfg_optimizer, self.model)
        return optimizer_ins.build_optimizer()

    def build_lr_scheduler(self, cfg_lr_scheduler):
        batch_size = self.train_batch_size
        train_epoch_size = self.train_epoch_size
        if self.iter_base:
            train_epoch_size = 1
            logger.info("using iter_base, data_size are set to 1")
        register_type = cfg_lr_scheduler.pop('lr_register_type', 'base')
        lr_scheduler_ins = LR_SCHEDULER_REGISTY[register_type](cfg_lr_scheduler,
                                                               self.optimizer,
                                                               train_epoch_size,
                                                               env.world_size * batch_size)

        return lr_scheduler_ins.build_scheduler()

    def _check_data_pool(self, pool, cfg_data):
        check_pool = []
        for p in pool:
            if p.split(':')[-1] in cfg_data:
                check_pool.append(p)
        return check_pool

    def build_dataloaders(self, reload_cfg=None):
        def build(builder_type, cfg_data, pool, reload_cfg):
            loaders = {}
            for idx, data_source in enumerate(pool):
                data_builder = DATA_BUILDER_REGISTY[builder_type[idx]](cfg_data, data_source, reload_cfg)
                source = data_source.split(':')[-1]
                loader = data_builder.build_dataloader()
                logger.info(f'build {data_source} done')
                loaders[source] = loader
            return loaders
        cfg_data = self.config['dataset']
        self.config['dataset']['builder_type'] = cfg_data.setdefault('builder_type', 'base')
        self.config['dataset']['data_ppol'] = cfg_data.setdefault('data_pool', ['train:train', 'test:test'])
        builder_type = cfg_data.get('builder_type', 'base')
        pool = cfg_data.get('data_pool', ['train:train', 'test:test'])
        pool = self._check_data_pool(pool, cfg_data)
        self.train_pool = [source for source in pool if source.startswith('train')]
        train_pool_idxs = [idx for idx in range(len(pool)) if pool[idx].startswith('train')]
        self.test_pool = [source for source in pool if source.startswith('test')]
        test_pool_idxs = [idx for idx in range(len(pool)) if pool[idx].startswith('test')]
        if not isinstance(builder_type, list):
            builder_type = [builder_type] * len(pool)
        if not self.training:
            pool = self.test_pool
            builder_type = [builder_type[idx] for idx in test_pool_idxs]
        self.data_loaders = build(builder_type, cfg_data, pool, reload_cfg)
        if 'train' in self.data_loaders:
            self.set_train_epoch_size()
            self.set_train_batch_size()

    def _remove_hooks(self, cfg_hooks):
        need_remove_hooks = ['auto_save_best', 'reload', 'auto_checkpoint']
        remove_set = set()
        for idx, hook in enumerate(cfg_hooks):
            for temp in need_remove_hooks:
                if temp in hook['type']:
                    remove_set.add(idx)
        for i in remove_set:
            del cfg_hooks[i]
        return cfg_hooks

    def build_global_flags(self):
        if self.aligned:
            ALIGNED_FLAG.aligned = True
            ALIGNED_FLAG.offset = 0
        if self.iter_base:
            ITER_BASE_FLAG.flag = True
        self.backend = DIST_BACKEND.backend

    def build_env(self):
        self.build_global_flags()
        rank_init = self.config['runtime']['rank_init']
        random_seed = self.config['runtime']['random_seed']
        set_random_seed(random_seed, rank_init)
        torch.backends.cudnn.enabled = not self.args.get('nocudnn', False)
        logger.info(f'world size:{env.world_size}')

    def resume_model(self):
        ckpt = self.saver.load_pretrain_or_resume()
        self.model.load(ckpt['model'])
        if self.training:
            if self.fp16:
                if self.backend == 'dist':
                    self.resume_fp16(ckpt)
                else:
                    self.optimizer.reload()
            if 'optimizer' in ckpt:
                start_epoch = ckpt['epoch']
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                logger.info(f'resume from epoch:{start_epoch} iteration:{self.cur_iter}')
        if 'ema' in ckpt and self.ema is not None:
            logger.info("load ckpt ema to model ema")
            self.ema.load_state_dict(ckpt['ema'])

    def build_saver(self):
        cfg_saver = self.config['saver']
        if 'kwargs' not in cfg_saver:
            _cfg_saver = {"type": 'base', "kwargs": {}}
            if 'type' in cfg_saver:
                _cfg_saver['type'] = cfg_saver.pop('type')
            _cfg_saver['kwargs']['save_cfg'] = cfg_saver
            _cfg_saver['kwargs']['yml_path'] = self.args.get('config_path', None)
            _cfg_saver['kwargs']['work_dir'] = self.work_dir
            saver = SAVER_REGISTRY.build(_cfg_saver)
            self.results_dir = cfg_saver.get('results_dir', 'results_dir')
        else:
            cfg_saver['kwargs']['yml_path'] = self.args.get('config_path', None)
            cfg_saver['kwargs']['work_dir'] = self.work_dir
            saver = SAVER_REGISTRY.build(cfg_saver)
            self.results_dir = cfg_saver['kwargs']['save_cfg'].get('results_dir', 'results_dir')
        self.saver = saver

    def get_git_version(self):
        try:
            root = os.path.dirname(__file__)
            root = os.environ.get('ROOT', root)
            cmd = f"cd {root} && git describe --tags"
            git_version = os.popen(cmd).read().strip()
        except:  # noqa
            git_version = ''
        logger.info("current git version {}".format(git_version))

    def upgrade_config(self, config):
        opts = config.get('args', {}).get('opts', [])
        config = merge_opts_into_cfg(opts, config)
        return config

    def build_ema(self):
        if self.config.get('ema', None) is None or not self.config['ema']['enable']:
            self.ema = None
            return
        ema_config = copy.deepcopy(self.config['ema'])
        ema_type = ema_config.get('ema_type', 'linear')
        ema_config['kwargs']['model'] = self.build_fake_model()
        self.ema = EMA_REGISTRY.get(ema_type)(**ema_config['kwargs'])

    def build_hooks(self):
        cfg_hooks = self.config.get('hooks', [])
        if not self.training:
            cfg_hooks = self._remove_hooks(cfg_hooks)
        self._hooks = build_hooks(self, cfg_hooks, add_log_if_not_exists=True)
        logger.info('build hooks done')

    def build_trainer(self):
        if not self.training:
            return
        cfg_trainer = self.config['trainer']
        self.optimizer = self.build_optimizer(cfg_trainer['optimizer'])
        self.lr_scheduler = self.build_lr_scheduler(cfg_trainer['lr_scheduler'])
        _freq_base = 1
        _start_base = 1
        if self.iter_base:
            _freq_base = 5000
        self.test_freq = cfg_trainer.get('test_freq', _freq_base)
        logger.info(f"test_freq: {self.test_freq}")
        self.test_start = cfg_trainer.get('test_start', _start_base)
        logger.info(f"test_start: {self.test_start}")
        self.save_freq = cfg_trainer.get('save_freq', _freq_base)
        logger.info(f"save_freq: {self.save_freq}")
        self.save_start = cfg_trainer.get('save_start', _start_base)
        self.only_save_latest = cfg_trainer.get('only_save_latest', False)
        logger.info(f'only save latest: {self.only_save_latest}')
        logger.info(f"save_start: {self.save_start}")
        max_epoch = cfg_trainer.get('max_epoch', 0)
        self.max_iter = cfg_trainer.get('max_iter', int(max_epoch * self.train_epoch_size))
        logger.info(f"max_iter: {self.max_iter}")

    def prepare_dist_model(self):
        if env.distributed:
            if self.backend == 'dist':
                logger.info('using ddp')
                self.model = DDP(self.model, device_ids=[env.local_rank], broadcast_buffers=False)
            else:
                self.model = DistModule(self.model, not self.args.get('asynchronize', False))

        if self.training:
            self.start_iter = self.lr_scheduler.last_iter
            for source in self.train_pool:
                dataset_name = source.split(':')[-1]
                self.data_loaders[dataset_name].batch_sampler.set_start_iter(self.start_iter)

    def save(self, auto_save=None, lns=None):
        save_dict = self.get_dump_dict()
        if auto_save is not None:
            save_dict['auto_save'] = auto_save
        if lns is not None:
            save_dict['lns'] = lns
        if env.is_master():
            self.saver.save(**save_dict)

    def save_epoch_ckpt(self, iter_idx):
        cur_iter = iter_idx + 1
        epoch_size = self.train_epoch_size
        if self.iter_base:
            epoch_size = 1
        assert not self.iter_base
        if cur_iter % epoch_size == 0 and iter_idx > self.start_iter:
            self.save(auto_save='latest', lns=False)

    def train(self):
        self.model.cuda().train()
        for iter_idx in range(self.start_iter, self.max_iter):
            batch = self.get_batch('train')
            loss = self.forward_train(batch)
            self.backward(loss)
            self.update()
            if self.ema is not None:
                self.ema.step(self.model, curr_step=iter_idx)
            if self.is_test(iter_idx):
                if iter_idx == self.start_iter:
                    continue
                if self.config['runtime']['async_norm']:
                    from eod.utils.env.dist_helper import all_reduce_norm
                    logger.info("all reduce norm")
                    all_reduce_norm(self.model)
                    if self.ema is not None:
                        all_reduce_norm(self.ema.model)
                self.evaluate()
                self.model.train()
            if self.only_save_latest:
                self.save_epoch_ckpt(iter_idx)
            else:
                if self.is_save(iter_idx):
                    self.save()
            self.lr_scheduler.step()

    @torch.no_grad()
    def _inference(self):
        self.model.cuda().eval()
        test_loader = self.data_loaders['test']
        all_results_list = []
        for _ in range(test_loader.get_epoch_size()):
            batch = self.get_batch('test')
            output = self.forward_eval(batch)
            dump_results = test_loader.dataset.dump(output)
            all_results_list.append(dump_results)
        barrier()
        all_device_results_list = all_gather(all_results_list)
        return all_device_results_list

    @torch.no_grad()
    def inference(self):
        all_device_results_list = self._inference()
        res_file = None
        if env.is_master():
            if not self.config['saver'].get('save_result', False):
                return res_file, all_device_results_list
            shutil.rmtree(self.results_dir, ignore_errors=True)
            os.makedirs(self.results_dir, exist_ok=True)
            res_file = os.path.join(self.results_dir, 'results.txt')
            logger.info(f'saving inference results into {res_file}')
            writer = open(res_file, 'w')
            for device_list in all_device_results_list:
                for results in device_list:
                    for item in results:
                        print(json.dumps(item), file=writer)
                        writer.flush()
            writer.close()
        return res_file, all_device_results_list

    @torch.no_grad()
    def evaluate(self):
        res_file, all_device_results_list = self.inference()
        if env.is_master():
            logger.info("begin evaluate")
            metrics = self.data_loaders['test'].dataset.evaluate(res_file, all_device_results_list)
            logger.info(json.dumps(metrics, indent=2))
        else:
            metrics = Metric({})
        barrier()
        self._hooks('after_eval', metrics)
        self.set_cur_eval_iter()
        return metrics

    def batch2device(self, batch):
        model_dtype = torch.float32
        if self.fp16 and self.backend == 'linklink':
            model_dtype = self.model.dtype
        if batch['image'].device != torch.device('cuda') or batch['image'].dtype != model_dtype:
            batch = to_device(batch, device=torch.device('cuda'), dtype=model_dtype)
        return batch

    def get_batch(self, batch_type='train'):
        if batch_type == 'train':
            cur_iter = self.cur_iter
        else:
            cur_iter = self.local_eval_iter(step=1)
        self._hooks('before_data', cur_iter)
        assert batch_type in self.data_loaders
        if not hasattr(self, 'data_iterators'):
            self.data_iterators = {}
        if batch_type not in self.data_iterators:
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
        else:
            iterator = self.data_iterators[batch_type]
        try:
            batch = next(iterator)
        except StopIteration as e:  # noqa
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
            batch = next(iterator)
        batch = dict(batch)
        batch = self.batch2device(batch)
        return batch

    def is_test(self, iter_idx):
        cur_iter = iter_idx + 1
        if cur_iter == self.max_iter:
            return True
        epoch_size = self.train_epoch_size
        if self.iter_base:
            epoch_size = 1
        if cur_iter >= self.test_start * epoch_size:
            if (cur_iter % (self.test_freq * epoch_size)) == 0:
                return True
        return False

    def is_save(self, iter_idx):
        cur_iter = iter_idx + 1
        if cur_iter == self.max_iter:
            return True
        epoch_size = self.train_epoch_size
        if self.iter_base:
            epoch_size = 1
        save_start = epoch_size * self.save_start
        save_fren = epoch_size * self.save_freq
        if cur_iter >= save_start and cur_iter % save_fren == 0:
            return True
        return False

    def forward_train(self, batch):
        assert self.model.training
        self._hooks('before_forward', self.cur_iter, batch)
        loss, output = self.forward(batch, return_output=True)
        self._hooks('after_forward', self.cur_iter, output)
        return loss

    def forward_eval(self, batch):
        self._hooks('before_eval_forward', self.local_eval_iter(), batch)
        assert not self.model.training
        output = self.forward_model(batch)
        self._hooks('after_eval_forward', self.local_eval_iter(), output)
        return output

    @property
    def local_iter(self):
        # index begin from 1
        return self.cur_iter % self.train_epoch_size + 1

    def cur_epoch(self):
        # epoch from 1
        epoch = (self.cur_iter // self.train_epoch_size) + 1
        return epoch

    @property
    def cur_iter(self):
        if hasattr(self, "lr_scheduler"):
            return self.lr_scheduler.last_iter

    def cur_eval_iter(self, step=None):
        if not hasattr(self, '_eval_iter'):
            self._eval_iter = 0
        if step is not None:
            self._eval_iter += step
        return self._eval_iter

    def set_cur_eval_iter(self):
        self._eval_iter = 0

    def local_eval_iter(self, step=None):
        return self.cur_eval_iter(step)

    def set_train_epoch_size(self):
        self.train_epoch_size = self.data_loaders['train'].get_epoch_size()

    def set_train_batch_size(self):
        self.train_batch_size = self.data_loaders['train'].batch_sampler.batch_size

    def special_bn_init(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                m.eps = 1e-3
                m.momentum = 0.03

    def build_model(self):
        model_helper_type = self.config['runtime']['model_helper']['type']
        model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
        model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
        self.model = model_helper_ins(self.config['net'], **model_helper_kwargs)
        if self.device == 'cuda':
            self.model = self.model.cuda()
        if self.config['runtime']['special_bn_init']:
            self.special_bn_init()
        if self.fp16 and self.backend == 'linklink':
            self.model = self.model.half()

    def build_fake_model(self):
        '''
        pt_sync_bn can't be deepcopy, replace with solo bn for ema
        '''
        net_cfg = self.config['net']
        normalize = {"type": "solo_bn"}
        flag = False
        for n in net_cfg:
            if 'normalize' in n['kwargs']:
                if n['kwargs']['normalize']['type'] == 'pt_sync_bn':
                    flag = True
                    n['kwargs']['normalize'] = normalize
        if flag:
            logger.info("pt_sync_bn can't be deepcopy, replace with solo bn for ema, \
                load model state to fake model state")
            model_helper_type = self.config['runtime']['model_helper']['type']
            model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
            model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
            model = model_helper_ins(net_cfg, **model_helper_kwargs)
            if self.device == 'cuda':
                model = model.cuda()
            if self.fp16 and self.backend == 'linklink':
                model = model.half()
            if self.config['runtime']['special_bn_init']:
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                        m.eps = 1e-3
                        m.momentum = 0.03
            model.load(self.model.state_dict())
        else:
            model = self.model
        return model

    @property
    def progress(self):
        progress = self.cur_iter / self.get_total_iter() * 100
        progress = min(max(0, progress), 100)
        if self._progress is not None:
            self._progress.set(progress)
        if self._eta is not None:
            t_avg = (time.time() - self._tik) / max(1.0, self.cur_iter - self.start_iter)
            eta = (self.get_total_iter() - self.cur_iter) * t_avg
            self._eta.set(eta)
            if self.cur_iter == self.start_iter:
                self._tik = time.time()
        return progress

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def resume_fp16(self, ckpt):
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])

    def get_fp16_dump_dict(self):
        return {'scaler': self.scaler.state_dict()}

    def get_dump_dict(self):
        model_dict = self.model.state_dict()
        dump_dict = {
            'epoch': self.cur_epoch(),
            'iter': self.cur_iter,
            'optimizer': self.optimizer.state_dict(),
            'model': model_dict,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        if self.fp16 and self.backend == 'dist':
            dump_dict.update(self.get_fp16_dump_dict())
        if self.ema is not None:
            dump_dict['ema'] = self.ema.state_dict()
        return dump_dict

    def forward_model(self, batch):
        if self.ema is not None and not self.model.training:
            return self.ema.model(batch)
        return self.model(batch)

    def forward(self, batch, return_output=False):
        if self.backend == 'dist' and self.fp16:
            with torch.cuda.amp.autocast(enabled=True):
                output = self.forward_model(batch)
        else:
            output = self.forward_model(batch)
        self._temporaries['last_output'] = output
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        loss = sum(losses)
        if return_output:
            return loss, output
        else:
            return loss

    def backward(self, loss):
        self.model.zero_grad()
        if self.backend == 'dist':
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        elif self.backend == 'linklink':
            loss = loss / env.world_size
            loss.backward()
            reduce_gradients(self.model, not self.args['asynchronize'],
                             self.args.get('allow_dead_parameter', False))
        else:
            raise NotImplementedError
        self._hooks('after_backward', self.cur_iter, loss)
        return loss

    def update(self):
        if self.backend == 'dist' and self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self._hooks('after_update', self.cur_iter)

    def get_total_iter(self):
        return self.max_iter
