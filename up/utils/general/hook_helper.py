# Standard Library
import json
import copy
import datetime
import os
import signal
import time
import weakref
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from .registry_factory import HOOK_REGISTRY, VISUALIZER_REGISTRY
from up.data.metrics.base_evaluator import Metric
from ..env.dist_helper import allreduce, env
from up.utils.env.analysis_utils import get_memory_info, split_node

# Import from local
from .log_helper import MetricLogger, SmoothedValue
from .log_helper import default_logger as logger
from .global_flag import (
    DIST_BACKEND,
    FP16_FLAG
)


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

    def before_train(self):
        """ Make sense before train
        """
        pass

    def after_train(self):
        """ Make sense after train
        """
        pass

    def before_allreduce(self, cur_iter):
        pass

    def after_allreduce(self, cur_iter):
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
        self.freq = runner.args.get('display', freq)
        SmoothedValue.skip_fist_k = skip_first_k
        if env.is_master():
            self.summary_writer = get_summary_writer_class(summary_writer)(os.path.join(runner.work_dir, logdir))
        self.train_timers = MetricLogger(delimiter=" ")
        self.eval_timers = MetricLogger(delimiter=" ")
        self.t_before_allreduce = 0
        self.t_after_allreduce = 0
        self.t_after_backward = 0
        self.t_after_forward = 0

    def before_data(self, cur_iter):
        self.t_before_data = time.time()

    def before_forward(self, cur_iter, input):
        self.t_after_data = time.time()

    def after_forward(self, cur_iter, output):
        self.t_after_forward = time.time()
        self.output = output

    def after_backward(self, cur_iter, loss):
        self.t_after_backward = time.time()

    def before_allreduce(self, cur_iter):
        # pass
        self.t_before_allreduce = time.time()

    def after_allreduce(self, cur_iter):
        self.t_after_allreduce = time.time()

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

        self.train_timers.update(
            cur_iter,
            data_time=self.t_after_data - self.t_before_data,
            batch_time=time.time() - self.t_before_data,
            forward_time=self.t_after_forward - self.t_after_data,
            backward_time=self.t_after_backward - self.t_after_forward,
            allreduce_time=self.t_after_allreduce - self.t_before_allreduce)

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
        self.eval_timers.update(cur_iter,
                                data_time=self.t_after_data - self.t_before_data,
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

    def upload_scalar(self, key, value, epoch):
        if isinstance(value, list):
            for idx, idx_value in enumerate(value) :
                self.summary_writer.add_scalar(key + f'_{idx}', idx_value, epoch)
        else:
            self.summary_writer.add_scalar(key, value, epoch)

    def after_eval(self, metrics):
        # if evaluate during training, record metrics into log
        runner = self.runner_ref()
        if env.is_master() and runner.training:
            epoch = runner.cur_epoch()
            for k, v in metrics.items():
                if not isinstance(v, list):
                    if isinstance(v, dict):
                        for i_k, i_v in v.items():
                            self.upload_scalar('metrics/' + k + '_' + i_k, i_v, epoch)
                    else:
                        self.upload_scalar('metrics/' + k, v, epoch)
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
        self.vis_gt = None
        self.vis_dt = None
        if vis_gt:
            vis_gt_dir = vis_gt['kwargs'].get('vis_dir', 'vis_gt')
            vis_gt['kwargs'].update({'vis_dir': vis_gt_dir})
            os.makedirs(vis_gt_dir, exist_ok=True)
            self.vis_gt = VISUALIZER_REGISTRY.build(vis_gt)
        if vis_dt:
            vis_dt_dir = vis_dt['kwargs'].get('vis_dir', 'vis_dt')
            vis_dt['kwargs'].update({'vis_dir': vis_dt_dir})
            os.makedirs(vis_dt_dir, exist_ok=True)
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


@HOOK_REGISTRY.register('feature_vis')
class FEATURE_VIS(Hook):
    """ Visualize ground truthes, detection results
    """

    def __init__(self, runner, layer_name, save_dir='feat_vis', gt_dir=None, interpolation='linear'):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(FEATURE_VIS, self).__init__(runner)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.show_with_gt = True
        if gt_dir is not None:
            self.show_with_gt = False
            self.gt_dir = gt_dir
            os.makedirs(gt_dir, exist_ok=True)

        interpolation_types = {'linear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST,
                               'cubic': cv2.INTER_CUBIC, 'area': cv2.INTER_AREA}
        self.interpolation = interpolation_types[interpolation]

        assert isinstance(layer_name, list), 'layer_name must be a layer name list'
        self.layer_name = layer_name
        self.feature = []

        def _get_features_hook(module, input, output):
            self.feature.append(output)
        for lyn in layer_name:
            vis_module = self.get_module_by_name(runner.model, lyn)
            vis_module.register_forward_hook(_get_features_hook)

        logger.info('build feature visualizer done')

    def get_module_by_name(self, model, name):
        name = name.split('.')
        targets_module = model
        for n in name:
            targets_module = getattr(targets_module, n)
        return targets_module

    def after_forward(self, cur_iter, output):
        assert len(self.feature) == len(self.layer_name), 'something wrong with the forward hook!'
        for bix in range(output['gt'].shape[0]):
            gt_save_once = True
            for fix, lyn in enumerate(self.layer_name):
                feature = self.feature[fix][bix].cpu().data.numpy()
                feature = np.sum(feature, axis=0)
                feature = np.maximum(feature, 0)
                feature -= np.min(feature)
                feature /= np.max(feature)
                feature = cv2.resize(feature, (224, 224), interpolation=self.interpolation)

                heatmap = cv2.applyColorMap(np.uint8(255 * feature), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]  # gbr to rgb

                heatmap = (heatmap * 255).astype(np.uint8)
                img = np.float32(output['image'][bix].cpu().numpy()) * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None]   # noqa
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

                filename = '_'.join(output['filenames'][bix].split('/'))
                sub_dir = os.path.join(self.save_dir, '_' + '_'.join(lyn.split('.')))
                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)
                if self.show_with_gt:
                    plt.figure()
                    plt.imshow(img)
                    plt.imshow(heatmap, alpha=0.5, cmap='rainbow')
                    plt.axis('off')
                    plt.savefig(os.path.join(sub_dir, filename))
                    plt.close()
                else:
                    if gt_save_once:
                        plt.figure()
                        plt.imshow(img)
                        plt.axis('off')
                        plt.savefig(os.path.join(self.gt_dir, filename))
                        plt.close()
                        gt_save_once = False
                    plt.figure()
                    plt.imshow(heatmap)
                    plt.axis('off')
                    plt.savefig(os.path.join(sub_dir, filename))
                    plt.close()
        self.feature = []


@HOOK_REGISTRY.register('grad_cam')
class Grad_CAM(Hook):
    """ Visualize ground truthes, detection results
    """

    def __init__(self, runner, layer_name=None, class_names=None, save_dir='grad_cam'):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(Grad_CAM, self).__init__(runner)
        if class_names is not None:
            with open(class_names, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = None
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if layer_name is None:
            cam_module = self.get_last_conv_module(runner.model)
        else:
            cam_module = self.get_module_by_name(runner.model, layer_name)
        self.feature = None
        self.gradient = None

        def _get_features_hook(module, input, output):
            self.feature = output

        def _get_grads_hook(module, input_grad, output_grad):
            self.gradient = output_grad[0]

        cam_module.register_forward_hook(_get_features_hook)
        cam_module.register_backward_hook(_get_grads_hook)

        logger.info('build grad cam visualizer done')
        logger.info('Please run grad cam in a training config!')

    def get_last_conv_module(self, model):
        targets_module = model
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                targets_module = m
        return targets_module

    def get_module_by_name(self, model, name):
        name = name.split('.')
        targets_module = model
        for n in name:
            targets_module = getattr(targets_module, n)
        return targets_module

    def before_forward(self, cur_iter, input):
        self.gradient = None
        self.feature = None
        runner = self.runner_ref()
        model = runner.model
        for bix in range(input['gt'].shape[0]):
            model.zero_grad()
            output = model(input)['logits']
            output[bix, input['gt'][bix]].backward()
            gradient = self.gradient[bix]
            weight = np.mean(gradient.cpu().numpy(), axis=(1, 2))
            feature = self.feature[bix].cpu().data.numpy()
            cam = feature * weight[:, np.newaxis, np.newaxis]
            cam = np.sum(cam, axis=0)
            cam = np.maximum(cam, 0)
            cam -= np.min(cam)
            cam /= np.max(cam)
            cam = cv2.resize(cam, (224, 224))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]  # gbr to rgb

            heatmap = (heatmap * 255).astype(np.uint8)
            img = np.float32(input['image'][bix].cpu().numpy()) * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None]    # noqa
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            filename = '_'.join(input['filenames'][bix].split('/'))
            plt.figure()
            if self.class_names is not None:
                raw_name = self.class_names[str(input['gt'][bix].item())]
                name, freq = raw_name.split('|')
                plt.title(name.strip())
                filename = freq.strip() + '_' + filename
            plt.imshow(img)
            plt.imshow(heatmap, alpha=0.5, cmap='rainbow')
            plt.axis('off')
            plt.savefig(os.path.join(self.save_dir, filename))

            runner.optimizer.step()
            self.gradient = None
            self.feature = None


@HOOK_REGISTRY.register('gradient_collector')
class GradientCollector(Hook):
    def __init__(self,
                 runner,
                 hook_module=["roi_head.cls_subnet_pred"],
                 collect_func_module="post_process.cls_loss",
                 grad_type='output'):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        super(GradientCollector, self).__init__(runner)
        self.grad_type = grad_type

        grad_collect_func_module = self.get_tar_module(runner.model, collect_func_module)
        collect_func = getattr(grad_collect_func_module, 'collect_grad')
        hook_target_modules = [self.get_tar_module(runner.model, t) for t in hook_module]

        def _backward_fn_hook(module, grad_in, grad_out):
            if self.grad_type == 'input':
                collect_func(grad_in[0])
            elif self.grad_type == 'output':
                collect_func(grad_out[0])
            else:
                raise NotImplementedError
        for m in hook_target_modules:
            m.register_full_backward_hook(_backward_fn_hook)

    def get_tar_module(self, model, targets):
        targets = targets.split('.')
        targets_module = model
        for t in targets:
            targets_module = getattr(targets_module, t)
        return targets_module


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
        # the grad in mixed precision is scaled, need unscale first
        if FP16_FLAG.fp16:
            optimizer = self.runner_ref().get_optimizer()
            if DIST_BACKEND.backend == 'dist':
                self.runner_ref().scaler.unscale_(optimizer)
            elif DIST_BACKEND.backend == 'linklink':
                if self.mode in [self._AVG, self._MAVG]:
                    raise ValueError('linklink fp16 only support grad cliping with pre_defined value!')
                optimizer.clip_grad_norm_(self.max_norm, self.norm_type)
                return

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
        if isinstance(metrics.v, list):
            value = sum(metrics.v) / len(metrics.v)
        else:
            value = metrics.v
        # save best ckpt
        if value > self.best_metric_val:
            self._save_best('best', value)
            self.best_metric_val = value
            self.best_epoch = self.runner_ref().cur_epoch()
        logger.info(f'best epoch: {self.best_epoch}; best val: {self.best_metric_val}')


@HOOK_REGISTRY.register('auto_save_best_multitask')
class AutoSaveBestMultiTask(AutoSaveBest):
    """
    """
    def __init__(self, runner, tasks=[]):
        super().__init__(runner)
        if len(tasks) == 0:
            tasks = runner._task_names  # save all
        self.tasks = tasks

        self.best_metrics = {}
        for task in self.tasks:
            self.best_metrics[task] = self.load_single_info(task)

    def load_single_info(self, task):
        ckpt_path = os.path.join(self.runner_ref().saver.save_dir, f'ckpt_best.{task}.pth')
        if os.path.exists(ckpt_path):
            ckpt = self.runner_ref().saver.load_checkpoint(ckpt_path)
        else:
            ckpt = {}
        best_metric_val = ckpt.get('metric_val', -1)
        best_epoch = ckpt.get('epoch', 0)
        return best_metric_val, best_epoch

    def after_eval(self, metrics):
        key_metrics = {}
        for m in metrics:
            if m.startswith('key.'):
                for task in self.tasks:
                    if m.startswith(f'key.{task}'):
                        key_metrics[task] = metrics[m]

        epoch = self.runner_ref().cur_epoch()
        for task, metric_val in key_metrics.items():
            if metric_val > self.best_metrics[task][0]:
                self._save_best(f'best.{task}', metric_val)

                self.best_metrics[task] = metric_val, epoch
            best_metric_val, best_epoch = self.best_metrics[task]
            logger.info(f'Task {task}: best epoch: {best_epoch}; best val: {best_metric_val}')


@HOOK_REGISTRY.register('auto_save_metric')
class AutoSaveMetric(Hook):
    def __init__(self, runner, metric_json='checkpoints/metric.json', detail_metric=''):
        super().__init__(runner)
        self.metric_json = metric_json
        self.detail_metric = detail_metric
        self.inference_only = not runner.training
        self._get_saved()

    def _get_saved(self):
        if os.path.exists(self.metric_json):
            try:
                self.metrics = json.load(open(self.metric_json))
            except:  # noqa
                self.metrics = []
        else:
            self.metrics = []
        return self.metrics

    def save_metric(self, js):
        self.metrics.append(js)
        d = os.path.dirname(self.metric_json)
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(self.metric_json, 'w') as fw:
            json.dump(self.metrics, fw, indent=2)

    def after_eval(self, metrics):
        metrics = dict(metrics)

        if env.is_master() and not self.inference_only:
            cur_epoch = self.runner_ref().cur_epoch()
            cur_iter = self.runner_ref().cur_iter
            js = {
                'metrics': metrics,
                'iter': cur_iter,
                'epoch': cur_epoch,
            }
            if os.path.exists(self.detail_metric):
                detail_metric = json.load(open(self.detail_metric))
                js['metrics'] = detail_metric
            self.save_metric(js)


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
            runner.build_dataloaders(reload_cfg=reload_cfg)
            runner.data_iterators['train'] = iter(runner.data_loaders["train"])


@HOOK_REGISTRY.register('memory_analysis')
class MemoryAnalysis(Hook):
    """Log time, loss, etc.
    """

    def __init__(self, runner, freq=20, logdir='log', summary_writer='tensorboard'):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
            - freq (:obj:`int`): frequency that output log messages
            - logdir (:obj:`str`): tensorboard events dir
            - summary_writer (:obj:`str`): "tensorboard" or "pavi"
        """
        super(MemoryAnalysis, self).__init__(runner)
        self.freq = runner.args.get('display', freq)
        node_list = split_node()
        logdir = logdir + '/mem/%d' % (env.rank // 8)
        if env.rank in node_list:
            self.summary_writer = get_summary_writer_class(summary_writer)(os.path.join(runner.work_dir, logdir))

    def after_update(self, cur_iter):
        cur_iter += 1
        if cur_iter % self.freq != 0:
            return
        node_info, node_list = get_memory_info(get_total=False)
        if env.rank in node_list:
            print(f"node memory info after iter {cur_iter}", node_info)
            for k, v in node_info.items():
                self.summary_writer.add_scalar('train/' + k, v, cur_iter)

    def after_eval_forward(self, cur_iter, output):
        cur_iter += 1
        if cur_iter % self.freq != 0:
            return
        node_info, node_list = get_memory_info(get_total=False)
        if env.rank in node_list:
            print(f"node memory info after iter {cur_iter}", node_info)
            for k, v in node_info.items():
                self.summary_writer.add_scalar('val/' + k, v, cur_iter)


@HOOK_REGISTRY.register('memory_empty')
class GPUMemoryEmpty(Hook):
    """Log time, loss, etc.
    """

    def __init__(self, runner, freq=100):
        super(GPUMemoryEmpty, self).__init__(runner)
        self.freq = freq

    def after_update(self, cur_iter):
        if cur_iter % self.freq == 0 and cur_iter > 0:
            torch.cuda.empty_cache()
            torch.cuda.memory.reset_peak_memory_stats()


@HOOK_REGISTRY.register('memory_checkpoint')
class MemoryCheckpoint(Hook):
    """Log time, loss, etc.
    """

    def __init__(self,
                 runner,
                 enable=True,
                 checkpoint_patterns={'backbone': {"patterns_mode": "level",
                                      "level": {'num': 'default'}}},
                 dc_cfg=None):
        super(MemoryCheckpoint, self).__init__(runner)
        self.enable = enable
        self.checkpoint_patterns = checkpoint_patterns
        self.checkpoint_set = set()
        self.dc_cfg = dc_cfg
        self.exclude_node = set()
        self.share_weight_node = {}

    def cast_forward(self):
        from .dc_manager import DynamicCheckpointManager, dc_cast_forward, common_cast_forward
        if self.dc_cfg is not None:
            self.dc_manager = DynamicCheckpointManager(self.dc_cfg.get("warmup_iters", 30),
                                                       self.checkpoint_set,
                                                       self.dc_cfg.get('max_memory', 8),
                                                       self.dc_cfg.get('debug_freq', -1),
                                                       self.dc_cfg.get('strategy', 'memory_time'),
                                                       self.share_weight_node)
        for name, m in self.runner_ref().model.named_modules():
            name = name.replace("module.", "")
            if name in self.checkpoint_set:
                if self.dc_cfg is not None:
                    dc_cast_forward(m, name, self.dc_manager)
                else:
                    common_cast_forward(m)

    def recover_forward(self):
        for name, m in self.runner_ref().model.named_modules():
            name = name.replace("module.", "")
            if name in self.checkpoint_set:
                if hasattr(m, "old_forward"):
                    m.forward = m.old_forward

    def before_forward(self, cur_iter, input):
        if self.enable:
            input['image'].requires_grad = True
            if self.dc_cfg is not None:
                image_shape = input['image'].shape
                self.dc_manager.input_size = image_shape[-2] * image_shape[-1]
                self.dc_manager.before_forward()

    def after_update(self, cur_iter):
        if self.enable and self.dc_cfg is not None:
            self.dc_manager.after_update()

    def is_leaf_node(self, cur_node, next_nodes):
        is_leaf = True
        for next_node in next_nodes:
            if next_node.startswith(cur_node):
                if next_node.count('.') > cur_node.count('.'):
                    is_leaf = False
        return is_leaf

    def get_bfs_checkpoint_node(self):
        checkpoint_keys = self.checkpoint_patterns.keys()
        for name, m in self.runner_ref().model.named_modules():
            name = name.replace("module.", "")
            if hasattr(m, "inplace") and m.inplace:
                self.exclude_node.add(name)
            for ck in checkpoint_keys:
                if name == ck or name.startswith(ck + '.'):
                    level = name.count('.')
                    max_level = max(self.checkpoint_patterns[ck].get('max_level', 0), level)
                    if level not in self.checkpoint_patterns[ck]:
                        self.checkpoint_patterns[ck][level] = []
                    self.checkpoint_patterns[ck][level].append(name)
                    self.checkpoint_patterns[ck]['max_level'] = max_level

    def add_leaf_node(self):
        for _, v in self.checkpoint_patterns.items():
            if v['patterns_mode'] == 'level':
                for level in range(1, v['max_level'] - 1):
                    for cur_node in v[level]:
                        if self.is_leaf_node(cur_node, v[level + 1]):
                            v[level + 1].append(cur_node)

    def _parse_node(self):
        for _, v in self.checkpoint_patterns.items():
            if v['patterns_mode'] == 'level':
                if v['level']['num'] == 'default':
                    level = v['max_level'] // 2
                else:
                    level = v['level']['num']
                assert level > 0, "level must > 0"
                num = len(v[level])
                for i in v[level][:num]:
                    self.checkpoint_set.add(i)
                    if v.get('share_weight_num', -1) > 0:
                        self.share_weight_node[i] = v['share_weight_num']
            elif v['patterns_mode'] == 'name':
                names = set(v.get('name_list', []))
                self.checkpoint_set = names
            else:
                logger.warning(f"patterns_mode {v['patterns_mode']} is not supported")
        for item in self.exclude_node:
            if item in self.checkpoint_set:
                self.checkpoint_set.remove(item)
        logger.info("checkpoint module name")
        logger.info(self.checkpoint_set)
        logger.info("share weight node: (name, share_num)")
        logger.info(self.share_weight_node)

    def parse_checkpoint_node(self):
        self.get_bfs_checkpoint_node()
        self.add_leaf_node()
        self._parse_node()

    def before_train(self):
        if self.enable:
            self.parse_checkpoint_node()
            self.cast_forward()

    def after_train(self):
        if self.enable:
            self.recover_forward()


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
