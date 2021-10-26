# Standard Library
import copy
import datetime
import os
import signal
import time
import weakref
from collections import defaultdict

import torch
from torch.nn.utils import clip_grad_norm_

from .registry_factory import HOOK_REGISTRY, VISUALIZER_REGISTRY
from eod.data.metrics.base_evaluator import Metric
from ..env.dist_helper import allreduce

# Import from local
from ..env.dist_helper import env
from .log_helper import MetricLogger, SmoothedValue
from .log_helper import default_logger as logger


__all__ = [
    'Hook',
    'TrainEvalLogger',
    'AutoCheckpoint',
    'GradClipper',
    'AutoSaveBest',
    'ReloadDataloader',
    'ComposeHook'
]


def get_summary_writer_class(_type='tensorboard'):
    SummaryWriter = None
    if _type == 'pavi':
        try:
            from pavi import SummaryWriter
        except ImportError:
            print('pavi not installed')

    if SummaryWriter is None:
        from tensorboardX import SummaryWriter
    return SummaryWriter


class Hook(object):
    """ A mechanism that decouples functions like log, time and visualization
    from training code.
    """

    def __init__(self, runner):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        self.runner_ref = weakref.ref(runner)

    def before_data(self, cur_iter):
        """ Make sense before fetching data from data loader
        Arguments:
          - cur_iter (int): current iteration in training
        """
        pass

    def before_forward(self, cur_iter, input):
        """ Make sense after fetching data and before forwarding
        Arguments:
          - cur_iter (int): current iteration in training
          - input (dict): input to model
        """
        pass

    def after_forward(self, cur_iter, output):
        """ Make sense after forwarding
        Arguments:
          - cur_iter (int): current iteration in training
          - output (dict): output of the model
        """
        pass

    def after_backward(self, cur_iter, loss):
        """ Make sense after backwarding
        Arguments:
          - cur_iter (int): current iteration in training
          - loss (tensor): training loss
        """
        pass

    def after_update(self, cur_iter):
        """ Make sense after updating params
        Arguments:
          - cur_iter (int): current iteration in training
        """
        pass

    def after_epoch(self, cur_epoch):
        """ Make sense after epoch
        Arguments:
          - cur_epoch (int): current training epoch
        """
        pass

    def before_eval_forward(self, cur_iter, input):
        """ Make sense before evaluation forward
        Arguments:
          - cur_iter (int): current training iteration
          - input (dict): input to model
        """
        pass

    def after_eval_forward(self, cur_iter, output):
        """ Make sense after evaluation forwarding
        Arguments:
          - cur_iter (int): current iteration in training
          - output (dict): output of the model
        """
        pass

    def after_eval(self, metrics):
        """ Make sense after evaluation
        Arguments:
          - metrics (dict): evaluation metrics
        """
        pass

    def __call__(self, hook_type, *args, **kwargs):
        fn = getattr(self, hook_type)
        return fn(*args, **kwargs)


@HOOK_REGISTRY.register('train_val_logger')
class TrainEvalLogger(Hook):
    """Log time, loss, etc.
    """

    def __init__(self, runner, freq=1, logdir='log', skip_first_k=1, summary_writer='tensorboard'):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
            - freq (:obj:`int`): frequency that output log messages
            - logdir (:obj:`str`): tensorboard events dir
            - skip_first_k (:obj:`int`): skip first several iteration when timing
            - summary_writer (:obj:`str`): "tensorboard" or "pavi"
        """
        super(TrainEvalLogger, self).__init__(runner)
        self.freq = freq
        SmoothedValue.skip_fist_k = skip_first_k
        if env.is_master():
            self.summary_writer = get_summary_writer_class(summary_writer)(os.path.join(runner.work_dir, logdir))
        self.train_timers = MetricLogger(delimiter=" ")
        self.eval_timers = MetricLogger(delimiter=" ")

    def before_data(self, cur_iter):
        self.t_before_data = time.time()

    def before_forward(self, cur_iter, input):
        self.t_after_data = time.time()

    def after_forward(self, cur_iter, output):
        self.output = output

    @classmethod
    def allreduce(cls, output):
        """Sync loss and accuracy across gpus.
           loss and accuracy must be torch.Tensor.
           return loss and accuracy only
        """

        def filter_fn(x):
            return x.find('loss') >= 0 or x.find('accuracy') >= 0

        output = {name: val.clone() for name, val in output.items() if filter_fn(name)}

        if env.distributed:
            for name, val in output.items():
                if torch.is_tensor(val):
                    allreduce(val)
                    output[name] = val / env.world_size

        return {name: val.item() for name, val in output.items()}

    @classmethod
    def get_loss(cls, output):
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        output['All.loss'] = sum(losses)
        # only loss and accuracy are kept
        output = cls.allreduce(output)

        losses = defaultdict(list)
        for name, value in output.items():
            prefix, local_name = name.split('.', 1)
            losses[prefix].append('{}:{:.4f}'.format(local_name, value))
        # merge prefix
        losses = sorted([
            "{}({})".format(prefix, " ".join(loss_list))
            for prefix, loss_list in losses.items()
        ])
        return output, " ".join(losses)

    def after_update(self, cur_iter):
        runner = self.runner_ref()
        cur_iter += 1
        total_iter = runner.get_total_iter()
        train_size = runner.train_epoch_size

        self.train_timers.update(data_time=self.t_after_data - self.t_before_data,
                                 batch_time=time.time() - self.t_before_data)

        eta_seconds = self.train_timers.batch_time.global_avg * (runner.get_total_iter() - cur_iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if cur_iter % self.freq != 0:
            return
        logger.info(
            " ".join([
                "Progress:[{progress:.1f}%][{cur_iter}/{total_iter}]",
                "Epoch:[{epoch}][{local_iter}/{epoch_len}]",
                "LR:{lr:.6f}",
                "Max mem:{memory:.0f}",
                "ETA:{eta}",
                "{timer}"
            ]).format(
                progress=runner.progress,
                cur_iter=cur_iter,
                total_iter=runner.get_total_iter(),
                epoch=runner.cur_epoch(),
                local_iter=runner.local_iter,
                epoch_len=min(train_size, (total_iter - 1) % train_size + 1),
                lr=runner.lr_scheduler.get_lr()[0],
                eta=eta_string,
                timer=self.train_timers,
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            ))
        output, loss_str = self.get_loss(self.output)
        del self.output
        if env.is_master():
            output['lr'] = runner.lr_scheduler.get_lr()[0]
            for k, v in output.items():
                self.summary_writer.add_scalar('train/' + k, v, cur_iter)
        logger.info(loss_str + '\n\n')

    def before_eval_forward(self, cur_iter, input):
        self.t_after_data = time.time()

    def after_eval_forward(self, cur_iter, output):
        self.eval_timers.update(data_time=self.t_after_data - self.t_before_data,
                                batch_time=time.time() - self.t_before_data)
        cur_iter += 1
        if cur_iter % self.freq != 0:
            return
        runner = self.runner_ref()
        test_len = runner.data_loaders['test'].get_epoch_size()
        eta_seconds = self.eval_timers.batch_time.global_avg * (test_len - cur_iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        logger.info(
            " ".join([
                "Progress:[{progress:.1f}%][{cur_iter}/{test_len}]",
                "Max mem:{memory:.0f}",
                "ETA:{eta}",
                "{timers}",
            ]).format(
                progress=min(cur_iter / test_len * 100, 100),
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                cur_iter=cur_iter,
                test_len=test_len,
                eta=eta_string,
                timers=self.eval_timers,
            ))

    def after_eval(self, metrics):
        # if evaluate during training, record metrics into log
        runner = self.runner_ref()
        if env.is_master() and runner.training:
            epoch = runner.cur_epoch()
            for k, v in metrics.items():
                if not isinstance(v, list):
                    self.summary_writer.add_scalar('metrics/' + k, v, epoch)
                    try:
                        self.summary_writer.flush()
                    except:  # noqa
                        pass


@HOOK_REGISTRY.register('visualize')
class Visualize(Hook):
    """ Visualize ground truthes, detection results
    """

    def __init__(self, runner, vis_gt=None, vis_dt=None):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
            - vis_gt: (:obj:`dict`): dict with keys {``output_dir``}
            - vis_dt: (:obj:`dict`): dict with keys {``output_dir``, ``bbox_thresh``}
        """
        super(Visualize, self).__init__(runner)
        vis_gt_dir = vis_gt['kwargs'].get('output_dir', 'vis_gt')
        vis_dt_dir = vis_dt['kwargs'].get('output_dir', 'vis_dt')
        os.makedirs(vis_gt_dir, exist_ok=True)
        os.makedirs(vis_dt_dir, exist_ok=True)
        # build visualizer
        self.vis_gt = VISUALIZER_REGISTRY.build(vis_gt)
        self.vis_dt = VISUALIZER_REGISTRY.build(vis_dt)
        logger.info('build visualizer done')

    def before_forward(self, cur_iter, input):
        if self.vis_gt:
            self.runner_ref().data_loaders['train'].dataset.vis_gt(self.vis_gt, input)

    def before_eval_forward(self, cur_iter, input):
        if self.vis_gt:
            self.runner_ref().data_loaders['test'].dataset.vis_gt(self.vis_gt, input)

    def after_eval_forward(self, cur_iter, output):
        if self.vis_dt:
            self.runner_ref().data_loaders['test'].dataset.vis_dt(self.vis_dt, output)


@HOOK_REGISTRY.register('auto_checkpoint')
class AutoCheckpoint(Hook):
    """Hook that used to take a snapshot automatically when killed
    """

    def __init__(self, runner):
        """
        Arguments
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(AutoCheckpoint, self).__init__(runner)
        dead_signals = ['SIGILL', 'SIGINT', 'SIGKILL', 'SIGQUIT', 'SIGSEGV', 'SIGSTOP', 'SIGTERM', 'SIGBUS']
        all_signals = dead_signals + ['SIGUSR1']

        def signal_handler(signal_num, frame):
            cur_epoch = runner.cur_epoch()
            cur_iter = runner.cur_iter
            if env.is_master():
                runner.saver.save(epoch=cur_epoch,
                                  iter=cur_iter,
                                  suffix='i{}'.format(cur_iter),
                                  state_dict=runner.model.state_dict(),
                                  optimizer=runner.optimizer.state_dict(),
                                  lr_scheduler=runner.lr_scheduler.state_dict())

            sig = signal.Signals(signal_num)
            if sig.name in dead_signals:
                logger.info('{}({}): oooops, someone is killing me! '
                            '@ epoch {}, iteration:{}'.format(sig.name, sig.value, cur_epoch, cur_iter))
                # sys.exit()
                exit(1)
            else:
                logger.info('{}({}): taking a snapshot! '
                            '@ epoch {}, iteration:{}'.format(sig.name, sig.value, cur_epoch, cur_iter))

        supervised_signals = []
        unsupported_signals = []
        for sig in all_signals:
            sig = getattr(signal, sig)
            try:
                signal.signal(sig, signal_handler)
                supervised_signals.append(sig)
            except Exception:
                unsupported_signals.append(sig)
        logger.info('supervised  signals:{}'.format(supervised_signals))
        logger.info('unsupported signals:{}'.format(unsupported_signals))


@HOOK_REGISTRY.register('grad_clipper')
class GradClipper(Hook):
    """
    Clip gradients by norm. The norm is determined by:
        - pre_defined mode: given max_norm
        - average mode: average accumulated norm from history
        - moving_average mode: moveing average accumulated norm from history
    """
    _PD = 'pre_defined'
    _AVG = 'average'
    _MAVG = 'moving_average'

    def __init__(self, runner, mode, max_norm=0, norm_type=2,
                 momentum=None, tolerance=1.0, display_freq=None):
        """
        Arguments:
          - runner (Runner): unified interface of runner
          - mode (str): choices are [pre_defined, average, moving_average]
          - max_norm (float):  max norm of the gradients
          - norm_type (float or int): type of the used p-norm. Can be 'inf' for infinity norm.
          - momentum (float): valid when mode == 'moving_average'
          - tolerance (float): tolerance factor applied to accumulated norm
          - display_freq (int): frequency to output accumulated norm
        """
        super(GradClipper, self).__init__(runner)
        assert mode in [self._PD, self._AVG, self._MAVG]
        if mode == self._PD:
            assert max_norm > 0
        elif mode == self._MAVG:
            assert momentum >= 0 and momentum <= 1

        self.mode = mode
        self.momentum = momentum
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.tolerance = tolerance
        self.display_freq = display_freq
        self.last_iter = 1

    def update_max_norm(self, parameters):
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type)
            total_norm += param_norm**self.norm_type
        total_norm = self.tolerance * total_norm**(1. / self.norm_type)
        if self.mode == self._AVG:
            factor = self.last_iter / (self.last_iter + 1)
        else:
            factor = self.momentum
        self.max_norm = self.max_norm * factor + (1 - factor) * total_norm
        self.last_iter += 1

    def after_backward(self, cur_iter, loss):
        model = self.runner_ref().get_model()
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if self.mode in [self._AVG, self._MAVG]:
            self.update_max_norm(parameters)
            if self.display_freq is not None and self.last_iter % self.display_freq == 0:
                logger.info('max_norm:{}'.format(self.max_norm))
        if torch.is_tensor(self.max_norm):
            for idx in range(len(parameters)):
                if parameters[idx].dtype != self.max_norm.dtype:
                    parameters[idx] = parameters[idx].to(dtype=self.max_norm.dtype)
        clip_grad_norm_(parameters, self.max_norm, self.norm_type)


@HOOK_REGISTRY.register('auto_save_best')
class AutoSaveBest(Hook):
    """Log time, loss, etc.
    """

    def __init__(self, runner):
        super(AutoSaveBest, self).__init__(runner)
        self.best_metric_val = -1
        self.best_epoch = 0
        # get past best ckpt metric val
        ckpt_path = os.path.join(self.runner_ref().saver.save_dir, 'ckpt_best.pth')
        if os.path.exists(ckpt_path):
            self.best_metric_val = self.runner_ref().saver.load_checkpoint(ckpt_path).get('metric_val', -1)
            self.best_epoch = self.runner_ref().saver.load_checkpoint(ckpt_path).get('epoch', 0)

    def _save_best(self, prefix='best', metric_val=0):
        cur_epoch = self.runner_ref().cur_epoch()
        cur_iter = self.runner_ref().cur_iter
        if env.is_master():
            if self.runner_ref().ema is not None:
                ema = self.runner_ref().ema.state_dict()
            else:
                ema = {}
            self.runner_ref().saver.save(epoch=cur_epoch,
                                         iter=cur_iter,
                                         lns=False,
                                         auto_save=prefix,
                                         metric_val=metric_val,
                                         state_dict=self.runner_ref().model.state_dict(),
                                         optimizer=self.runner_ref().optimizer.state_dict(),
                                         ema=ema,
                                         lr_scheduler=self.runner_ref().lr_scheduler.state_dict())

    def after_eval(self, metrics):
        if not isinstance(metrics, Metric):
            logger.info("auto save best checkpoint only support Metric !!!")
            return
        # save best ckpt
        if metrics.v > self.best_metric_val:
            self._save_best('best', metrics.v)
            self.best_metric_val = metrics.v
            self.best_epoch = self.runner_ref().cur_epoch()
        logger.info(f'best epoch: {self.best_epoch}; best val: {self.best_metric_val}')


@HOOK_REGISTRY.register('reload')
class ReloadDataloader(Hook):
    def __init__(self, runner, group=1):
        super(ReloadDataloader, self).__init__(runner)
        self.group = group

    def after_update(self, cur_iter):
        runner = self.runner_ref()
        cur_iter += 1
        if not hasattr(runner.data_loaders['train'].dataset, 'mini_epoch'):
            return
        mini_epoch_size = runner.data_loaders['train'].get_epoch_size()
        mini_epoch = runner.data_loaders['train'].dataset.mini_epoch
        total_size = mini_epoch_size * mini_epoch
        if cur_iter > 0 and cur_iter % mini_epoch_size == 0:
            mini_epoch_idx = cur_iter // mini_epoch_size % mini_epoch
            epoch = cur_iter // total_size
            reload_cfg = {
                "seed": epoch + 1000,
                "mini_epoch": mini_epoch,
                "mini_epoch_idx": mini_epoch_idx,
                "group": self.group
            }
            del runner.data_loaders, runner.data_iterators['train']
            import gc
            gc.collect()
            if not hasattr(self, 'data_iterators'):
                runner.data_iterators = {}
            runner.data_loaders = runner.build_dataloaders(reload_cfg=reload_cfg)
            runner.data_iterators['train'] = iter(runner.data_loaders["train"])


class ComposeHook(object):
    """A interface that compose several hooks into one
    """

    def __init__(self, hooks):
        self.hooks = hooks

    def __call__(self, *args, **kwargs):
        results = [hook(*args, **kwargs) for hook in self.hooks]
        return results


def build_hooks(runner, cfg_list, add_log_if_not_exists=True):

    def add_log_hook(cfg_hooks):
        exists = any(['train_val' in cfg['type'] for cfg in cfg_hooks])
        if not exists:
            cfg_hooks.insert(0, {
                'type': 'train_val_logger',
                'kwargs': {
                    'freq': runner.args.get('display', 10),
                    'skip_first_k': 5,
                    'logdir': 'log'
                }
            })
        return cfg_hooks

    def build_single_hook(cfg):
        cfg = copy.deepcopy(cfg)
        kwargs = cfg.setdefault('kwargs', {})
        kwargs['runner'] = runner
        return HOOK_REGISTRY.build(cfg)

    if add_log_if_not_exists:
        cfg_list = add_log_hook(cfg_list)

    hooks = [build_single_hook(cfg) for cfg in cfg_list]
    return ComposeHook(hooks)
