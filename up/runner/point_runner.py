import torch
import torch.optim
import os
import shutil
import json
import numpy as np
from up.utils.general.log_helper import default_logger as logger
from up.utils.env.gene_env import to_device
from up.utils.env.dist_helper import barrier, all_gather, env
from up.data.metrics.base_evaluator import Metric
from .base_runner import BaseRunner
from up.utils.general.registry_factory import RUNNER_REGISTRY
import pathlib
try:
    import kornia
except BaseException:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


__all__ = ['PointRunner']


@RUNNER_REGISTRY.register("point")
class PointRunner(BaseRunner):
    def __init__(self, config, work_dir='./', training=True):
        super(PointRunner, self).__init__(config, work_dir, training)

    def batch2device(self, batch):
        model_dtype = torch.float32
        if self.fp16 and self.backend == 'linklink':
            model_dtype = self.model.dtype
        for key, val in batch.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib', 'voxel_infos', 'class_names']:
                continue
            elif key in ['images']:
                batch[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch[key] = torch.from_numpy(val).int().cuda()
            else:
                batch[key] = torch.from_numpy(val).float().cuda()

        if batch['points'].device != torch.device('cuda') or batch['points'].dtype != model_dtype:
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

    def statistics_info(self, recall_thresh_list, ret_dict, metric, disp_dict):
        for cur_thresh in recall_thresh_list:
            metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
            metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
        metric['gt_num'] += ret_dict.get('gt', 0)
        min_thresh = recall_thresh_list[0]
        disp_dict['recall_%s' % str(min_thresh)] = \
            '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)],
                               metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
        return metric, disp_dict

    @torch.no_grad()
    def _inference(self):
        self.model.cuda().eval()
        test_loader = self.data_loaders['test']
        all_results_list = []
        # generate_prediction_dicts 设置成静态函数所以参数只能这么传进来
        self.class_names = test_loader.dataset.class_names
        self.kitti_infos = test_loader.dataset.kitti_infos
        res_size = len(test_loader.dataset)
        self.recall_thresh_list = test_loader.dataset.evaluator.recall_thresh_list
        self.final_output_dir = os.path.join(self.results_dir, 'final_result', 'data')
        pathlib.Path(self.final_output_dir).mkdir(parents=True, exist_ok=True)
        # metric
        metric = {
            'gt_num': 0,
        }
        for cur_thresh in self.recall_thresh_list:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0
        # 循环
        for _ in range(test_loader.get_epoch_size()):
            batch = self.get_batch('test')
            results = self.forward_eval(batch)
            disp_dict = {}
            metric, disp_dict = self.statistics_info(self.recall_thresh_list, results, metric, disp_dict)
            dump_results = test_loader.dataset.generate_prediction_dicts(
                batch, results['pred_dicts'], self.class_names, self.final_output_dir)
            all_results_list.append(dump_results)
        barrier()
        part_list = all_gather(all_results_list)
        all_device_results_list = []
        for res_gpus in zip(*part_list):  # [8,114,4]-[114个[8,4]]
            for idx in range(len(res_gpus[-1])):
                res_bs = [res[idx] for res in res_gpus]
                all_device_results_list.extend(res_bs)
        for res_gpu in res_gpus:
            if len(res_gpu) > len(res_gpus[-1]):
                all_device_results_list.extend([res_gpu[-1]])
        metric_list = all_gather(metric)
        return all_device_results_list[:res_size], metric_list

    @torch.no_grad()
    def inference(self):
        all_device_results_list, metric = self._inference()
        # metric
        if env.distributed:
            for key, val in metric[0].items():
                for k in range(1, env.world_size):
                    metric[0][key] += metric[k][key]
        metric = metric[0]
        gt_num_cnt = metric['gt_num']
        ret_dict = {}
        for cur_thresh in self.recall_thresh_list:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        # save all_device_results_list
        res_file = None
        if env.is_master():
            if not self.config['saver'].get('save_result', False):
                return res_file, all_device_results_list, ret_dict
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
        return res_file, all_device_results_list, ret_dict

    """
    @torch.no_grad()
    def evaluate(self):
        if env.is_master():
            return self.evaluate_single()
    """
    @torch.no_grad()
    def evaluate(self):
        res_file, all_device_results_list, ret_dict = self.inference()
        if env.is_master():
            logger.info("begin evaluate")
            # eval_metric = self.config.get('evaluator', {}).get('type', None)
            result, ret_dict = self.data_loaders['test'].dataset.evaluate(
                res_file, self.class_names, self.kitti_infos, all_device_results_list)
            metrics = ret_dict
            logger.info(result)
            logger.info(json.dumps(ret_dict, indent=2))
        else:
            metrics = Metric({})
        barrier()
        self._hooks('after_eval', metrics)
        self.set_cur_eval_iter()
        return metrics
