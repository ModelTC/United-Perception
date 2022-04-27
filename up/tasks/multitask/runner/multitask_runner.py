import os
import torch
import json

from up.data.metrics.base_evaluator import Metric
from up.runner.base_runner import BaseRunner
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.dist_helper import env, barrier
from up.utils.general.registry_factory import RUNNER_REGISTRY


@RUNNER_REGISTRY.register('multitask')
class MultiTaskRunner(BaseRunner):
    def build(self):
        self.init_multitask()
        super().build()

    def init_multitask(self):
        cfg = self.config.get('multitask_cfg')

        # dataset
        self._train_datasets = cfg['datasets']['train']
        self._test_datasets = cfg['datasets']['test']

        if 'data_pool' not in self.config['dataset']:
            dataset = self.config['dataset']
            train_pool = [f'train:{name}' for name in self._train_datasets]
            test_pool = [f'test:{name}' for name in self._test_datasets]

            dataset['data_pool'] = train_pool + test_pool

            logger.warn('train_pool default set to {}'.format(
                dataset['data_pool']))

        # model idx
        self._train_model_idxs = cfg.get('model_idxs', {}).get('train', None)
        self._test_model_idxs = cfg.get('model_idxs', {}).get('test', None)

        # task names
        self._task_names = cfg['task_names']

        self._multitask_debug = cfg.get('debug', False)

    def _set_task_idx(self, model, model_idx, task_idx=None):
        # model_idx: split bn
        # task_idx: module idx, to split data
        if self._multitask_debug:
            logger.info(f'model_idx: {model_idx}, task_idx: {task_idx}')
        for _, m in model.named_modules():
            if hasattr(m, 'set_task_idx'):
                m.set_task_idx(model_idx)

            # module
            if task_idx and hasattr(m, '_multitask_forward'):
                m.set_task_idx(task_idx)

            if self._multitask_debug and hasattr(m, 'set_task_idx'):
                logger.info(f'{_}: {m.task_idx}')

    def set_task_idx(self, model_idx, task_idx=None):
        if self._multitask_debug:
            logger.info('setting default model')

        self._set_task_idx(self.model, model_idx, task_idx)
        if self.ema is not None:
            if self._multitask_debug:
                logger.info('setting ema model')
            self._set_task_idx(self.ema.model, model_idx, task_idx)

    def forward_train_single_task(self, batch, idx):
        if self._train_model_idxs:
            task_idx = idx
            model_idx = self._train_model_idxs[idx]
        else:
            task_idx = None
            model_idx = idx

        self.set_task_idx(model_idx, task_idx)

        output = self.model(batch)
        losses = [
            val for name, val in output.items() if name.find('loss') >= 0
        ]
        loss = sum(losses)
        return loss, output

    def forward_train(self, batches):
        assert self.model.training

        output = {}
        losses = []

        self._hooks('before_forward', self.cur_iter, batches[0])

        for idx, batch in enumerate(batches):
            loss, out = self.forward_train_single_task(batch, idx)
            output.update(out)
            losses.append(loss)

        self._hooks('after_forward', self.cur_iter, output)
        return losses

    def _get_train_batches(self):
        batches = []
        for name in self._train_datasets:
            batch = self.get_batch(name)
            batches.append(batch)
        return batches

    def _get_eval_params(self):
        if self._test_model_idxs is None:
            for task_idx, (task_name, data_name) in enumerate(
                    zip(self._task_names, self._test_datasets)):
                yield task_idx, task_name, data_name, None
        else:
            for task_idx, (task_name, data_name, model_idx) in enumerate(
                    zip(self._task_names, self._test_datasets,
                        self._test_model_idxs)):
                yield model_idx, task_name, data_name, task_idx

    @torch.no_grad()
    def sub_evaluate(self):
        res_file, all_device_results_list = self.inference()
        if env.is_master():
            logger.info("begin evaluate")
            metrics = self.data_loaders['test'].dataset.evaluate(
                res_file, all_device_results_list)
            logger.info(json.dumps(metrics, indent=2))
        else:
            metrics = Metric({})
        barrier()

        # self._hooks('after_eval', metrics)
        self.set_cur_eval_iter()
        return metrics

    def evaluate(self):
        total_results_dir = self.results_dir
        metrics = None
        self._multitask_metrics = {}

        for model_idx, task_name, data_name, task_idx in self._get_eval_params(
        ):
            info = f'eval {task_name} with {data_name}, '
            if task_idx:
                info += f'task idx is {task_idx}, model idx is {model_idx}'
            else:
                info += f'task idx is {model_idx}'
            logger.info(info)
            if data_name == '':
                logger.warn(f'ignore test for {task_name}')
                continue
            # set model
            self.set_task_idx(model_idx, task_idx)
            # set result dir
            self.results_dir = os.path.join(total_results_dir, task_name)
            # set data
            backup_loader = self.data_loaders.get('test', None)
            target = self.data_loaders[data_name]
            self.data_loaders['test'] = target

            # force reset data_iterators
            if hasattr(self, 'data_iterators'):
                if 'test' in self.data_iterators:
                    del self.data_iterators['test']

            # eval: only use hook once
            # m = super().evaluate()
            m = self.sub_evaluate()

            # rename
            m.update({f'all.{task_name}.{k}': v for k, v in m.items()})

            # update metric
            if metrics is None:
                metrics = m
            else:
                metrics.update(m)

            # save key metric
            self._multitask_metrics[task_name] = {m.cmp_key: m.v}
            metrics[f'key.{task_name}.{m.cmp_key}'] = m.v

            # restore loader
            if backup_loader is not None:
                self.data_loaders['test'] = backup_loader

        # restore dir
        self.results_dir = total_results_dir
        logger.info(json.dumps(metrics, indent=2))
        self._hooks('after_eval', metrics)
        return metrics

    def train(self):
        self.model.cuda().train()

        for iter_idx in range(self.start_iter, self.max_iter):
            batches = self._get_train_batches()
            losses = self.forward_train(batches)
            if self.backend == 'dist':
                self.model.zero_grad()
                fake_loss = 0.0
                for param in self.model.parameters():
                    fake_loss += param.sum() * 0
                for loss in losses:
                    loss += fake_loss
                    if self.fp16:
                        self.scaler.scale(loss).backward(retain_graph=True)
                    else:
                        loss.backward(retain_graph=True)
                self._hooks('after_backward', self.cur_iter, sum(losses))
            else:
                loss = sum(losses)
                self.backward(loss)
            self.update()

            # EMA
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
                self.evaluate()
                self.model.train()
            if self.only_save_latest:
                self.save_epoch_ckpt(iter_idx)
            else:
                if self.is_save(iter_idx):
                    self.save()
            self.lr_scheduler.step()
