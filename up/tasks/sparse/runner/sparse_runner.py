from up.utils.general.log_helper import default_logger as logger
from up.utils.general.cfg_helper import format_cfg
from up.utils.env.gene_env import get_env_info, print_network
from up.utils.env.dist_helper import env, DistModule
from torch.nn.parallel import DistributedDataParallel as DDP
from up.utils.general.registry_factory import (
    RUNNER_REGISTRY,
    MODEL_HELPER_REGISTRY
)
from up.runner.base_runner import BaseRunner

__all__ = ['SparseRunner']


@RUNNER_REGISTRY.register('sparse')
class SparseRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        super(SparseRunner, self).__init__(config, work_dir, training)

    def build(self):
        self.build_env()
        self.get_git_version()
        self.build_dataloaders()
        self.prepare_fp16()
        self.build_model()
        self.build_saver()
        self.build_hooks()
        self.build_ema()
        self.load_ckpt()

        self.resume_model_from_fp()
        self.prepare_sparse_model()
        self.build_trainer()
        self.resume_model_from_sparse()
        self.sparse_post_process()

        self.prepare_dist_model()
        get_env_info()
        self.save_running_cfg()
        if not self.args.get('no_running_config', False):
            logger.info('Running with config:\n{}'.format(format_cfg(self.config)))

    def build_ema(self):
        if self.config.get('ema', None) and self.config['ema']['enable']:
            logger.warning("Don't use ema technique during the  sparsification-aware training!")
            logger.warning("If ema is used, it will be ignored during training!")
        self.ema = None

    def sparse_post_process(self):
        if self.config["sparsity"]["scheduler"]["type"] == "AmbaLevelPruneScheduler":
            self.sparse_scheduler.set_lr_scheduler(self.lr_scheduler)
        elif self.config["sparsity"]["scheduler"]["type"] == "AmpereScheduler":
            self.sparse_scheduler.init_optimizer(self.optimizer)

    def load_ckpt(self):
        self.ckpt = self.saver.load_pretrain_or_resume()
        if 'ema' in self.ckpt:
            if "ema_state_dict" in self.ckpt['ema']:
                logger.info("load ckpt from ema state_dict ...")
                self.ckpt['model'] = self.ckpt['ema']['ema_state_dict']

    def resume_model_from_fp(self):
        if 'model' in self.ckpt:
            self.model.load(self.ckpt['model'])

    def resume_model_from_sparse(self):
        if self.training and 'optimizer' in self.ckpt:
            start_epoch = self.ckpt['epoch']
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(self.ckpt['lr_scheduler'])
            self.sparse_scheduler.iteration = self.ckpt['iter']
            logger.info(f'resume from epoch:{start_epoch} iteration:{self.cur_iter}')

    def prepare_dist_model(self):
        if env.distributed:
            if self.backend == 'dist':
                logger.info('using ddp')
                self.model = DDP(self.model, device_ids=[env.local_rank], broadcast_buffers=False)
            else:
                self.model = DistModule(self.model, not self.args.get('asynchronize', False))

        if self.training:
            self.start_iter = self.cur_iter         # NOTE： use self.cur_iter, don't use self.lr_scheduler.last_iter
            for source in self.train_pool:
                dataset_name = source.split(':')[-1]
                self.data_loaders[dataset_name].batch_sampler.set_start_iter(self.start_iter)

    def train(self):
        self.model.cuda().train()
        for iter_idx in range(self.start_iter, self.max_iter):
            batch = self.get_batch('train')
            loss = self.forward_train(batch)
            self.backward(loss)
            self.update()
            self.sparse_scheduler.prune_model(self.model, iter_idx)
            if self.ema is not None:
                self.ema.step(self.model, curr_step=iter_idx)
            if self.is_test(iter_idx):
                if iter_idx == self.start_iter:
                    continue
                if self.config['runtime']['async_norm']:
                    from up.utils.env.dist_helper import all_reduce_norm
                    logger.info("all reduce norm")
                    all_reduce_norm(self.model)
                    if self.ema is not None:
                        all_reduce_norm(self.ema.model)
                logger.info("current sparse rate = {}".format(self.sparse_scheduler.compute_sparse_rate(self.model)))
                self.evaluate()
                self.model.train()
            if self.only_save_latest:
                self.save_epoch_ckpt(iter_idx)
            else:
                if self.is_save(iter_idx):
                    self.save()
            self.lr_scheduler.step()

    def build_model(self):
        SparseModelHelper = MODEL_HELPER_REGISTRY['sparse']
        self.model = SparseModelHelper(self.config)
        self.model_list = self.model.model_list
        self.model_map = self.model.model_map
        if env.is_master():
            logger.debug(self.model)
            logger.debug(self.model_list)
            logger.debug(self.model_map)
        if self.device == 'cuda':
            self.model = self.model.cuda()
        if self.config['runtime']['special_bn_init']:
            self.special_bn_init()
        if self.fp16 and self.backend == 'linklink':
            self.model = self.model.half()
        logger.debug(print_network(self.model))

    def get_leaf_module(self, leaf_module):
        from msbench.nn.modules import FrozenBatchNorm2d
        from up.tasks.det.plugins.yolov5.models.components import Space2Depth
        leaf_module_map = {'FrozenBatchNorm2d': FrozenBatchNorm2d,
                           'Space2Depth': Space2Depth}
        leaf_module_ret = [leaf_module_map[k] for k in leaf_module]
        return tuple(leaf_module_ret)

    def prepare_sparse_model(self):
        cfg_trainer = self.config['trainer']
        import copy
        prepare_custom_config_dict = copy.deepcopy(self.config["sparsity"])

        max_epoch = cfg_trainer.get('max_epoch', 0)
        max_iter = cfg_trainer.get('max_iter', int(max_epoch * self.train_epoch_size))
        if prepare_custom_config_dict["scheduler"]["type"] == 'AmbaLevelPruneScheduler':
            prepare_custom_config_dict["scheduler"]["kwargs"]["total_iters"] = max_iter
            default_input_custom = self.get_batch("train")
            prepare_custom_config_dict["scheduler"]["kwargs"]["default_input_custom"] = default_input_custom

        leaf_module = prepare_custom_config_dict.get('leaf_module', None)
        if leaf_module is not None:
            prepare_custom_config_dict["leaf_module"] = self.get_leaf_module(leaf_module)

        from msbench.scheduler import build_sparse_scheduler
        self.sparse_scheduler = build_sparse_scheduler(prepare_custom_config_dict)
        self.model.train()
        if prepare_custom_config_dict["scheduler"]["type"] == 'AmpereScheduler':
            self.model = self.sparse_scheduler.prepare_sparse_model(model=self.model)
        else:
            for mname in self.model_list:
                mod = getattr(self.model, mname)
                mod = self.sparse_scheduler.prepare_sparse_model(model=mod)
                setattr(self.model, mname, mod)
                if env.is_master():
                    logger.info(mod)
        self.model.train()
        if env.is_master():
            logger.info(self.model)

    @property
    def cur_iter(self):
        if hasattr(self, "lr_scheduler"):
            if hasattr(self.sparse_scheduler, "get_last_prune_step"):
                return self.lr_scheduler.last_iter + self.sparse_scheduler.get_last_prune_step()
            else:
                return self.lr_scheduler.last_iter

    def is_save(self, iter_idx):
        cur_iter = iter_idx + 1
        if cur_iter == self.max_iter:
            # apply mask to layer's weight, and disable the fake sparsification process
            self.sparse_scheduler.export_sparse_model(self.model)
            return True
        epoch_size = self.train_epoch_size
        if self.iter_base:
            epoch_size = 1
        save_start = epoch_size * self.save_start
        save_fren = epoch_size * self.save_freq
        if cur_iter >= save_start and cur_iter % save_fren == 0:
            return True
        return False
