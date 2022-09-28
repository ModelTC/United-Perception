import copy
import time
import json
import itertools
import random

import torch
import torch.nn.functional as F
from easydict import EasyDict

from up.utils.env.dist_helper import broadcast, env
from up.utils.env.dist_helper import DistModule
from up.utils.env.gene_env import to_device
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.global_flag import DIST_BACKEND
from up.utils.general.saver_helper import Saver
from up.tasks.distill.mimicker import Base_Mimicker

from up.tasks.nas.bignas.controller.utils.misc import count_dynamic_flops_and_params, \
    get_kwargs_itertools, clever_format, parse_flops, clever_dump_table, get_image_size_with_shape, \
    reset_model_bn_forward, copy_bn_statistics, rewrite_batch_norm_bias, adapt_settings

from up.tasks.nas.bignas.models.search_space import BignasSearchSpace


class BaseController(object):
    def __init__(self, config, cfg=None, model=None, teacher_models=None):
        self.config = EasyDict(config)
        self.build_bignas_settings()
        self.cfg = cfg
        self.model = model
        self.teacher_models = teacher_models
        self.build_subnet_table()
        self.init_distiller()

    def set_path(self, path):
        self.path = path

    def build_bignas_settings(self):
        # data
        self.metric1 = self.config.data.get('metric1', 'top1')
        self.metric2 = self.config.data.get('metric2', 'top5')

        # make sure image_size_list is in ascending order
        self.image_size_list = self.config.data.image_size_list

        self.share_input = self.config.data.get('share_input', False)
        self.interpolation_type = self.config.data.get('interpolation_type', 'bicubic')
        assert self.interpolation_type in [
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', None]

        self.calib_meta_file = self.config.data.get(
            'calib_meta_file', '/mnt/lustre/share/shenmingzhu/train_4k.txt')

        # train
        self.valid_before_train = self.config.train.get(
            'valid_before_train', False)
        self.sample_subnet_num = self.config.train.get('sample_subnet_num', 1)
        self.sample_strategy = self.config.train.get(
            'sample_strategy', ['max', 'random', 'random', 'min'])

        # distiller
        self.distiller = self.config.get('distiller', None)
        if self.distiller:
            self.distiller_type = self.config.distiller.get('type', 'kd')

        # subnet
        self.subnet = self.config.get('subnet', None)

        # latency
        self.latency = self.config.get('latency', None)

    def build_subnet_table(self):
        self.subnet_table = {}
        for i in self.sample_strategy:
            if isinstance(i, int):
                self.subnet_table = self.get_subnet_num(count_flops=True, print_log=True)
                break

    def init_distiller(self):
        if not self.config.get('distiller', False):
            return
        self.mimic_configs, self.teacher_configs = self.prepare_mimic_configs(self.config['distiller']['mimic'])
        for tmk in self.teacher_models.keys():
            if self.config['distiller'][tmk]:
                self.load_teacher_weight(tmk, self.teacher_models[tmk],
                                         self.teacher_configs[tmk].get('teacher_weight', None),
                                         self.teacher_configs[tmk].get('teacher_bn_mode', None))
        self.teacher_output_dict = {}
        for key in self.mimic_configs.keys():
            self.teacher_output_dict[key] = {}
        self.build_mimic()

    def prepare_mimic_configs(self, raw_configs):
        self.task_loss = True
        if 'task_loss' in raw_configs:
            self.task_loss = raw_configs.pop('task_loss')
        mimic_configs = {}
        if 'mimic_as_jobs' in raw_configs:
            assert raw_configs['mimic_as_jobs'] is True, 'mimic_as_jobs must be True or do not appear!'
            raw_configs.pop('mimic_as_jobs')
            mimic_configs = raw_configs
        else:
            assert 'student' in raw_configs.keys(), 'Wrong mimic setting! Single mimic job must contain the student key. \
                If you want to mimic multi jobs, set the mimic_as_jobs=True and name each mimic as a mimic_job'
            mimic_configs['mimic_job0'] = raw_configs
        teacher_configs = {}

        for mjx in mimic_configs.keys():
            assert 'mimic_job' in mjx, 'mimic jobs must start with the mimic_job prefix!'
            mimic_configs[mjx]['t_names'] = []
            for atk in mimic_configs[mjx].keys():
                if 'teacher' in atk:
                    assert atk not in teacher_configs, 'The teacher keys are unique even in different mimic jobs. \
                        If you want to use a same teacher model with multi mimic methods, \
                        you must copy and rename it to multi teacher model configs.'
                    teacher_configs.update({atk: mimic_configs[mjx][atk]})
                    mimic_configs[mjx]['t_names'].append(atk)
            assert len(mimic_configs[mjx]['t_names']) > 0, 'useless mimic job {}'.format(mjx)
        no_config_teachers = list(set(self.teacher_models.keys()) - set(teacher_configs.keys()))
        extra_config_teachers = list(set(teacher_configs.keys()) - set(self.teacher_models.keys()))
        if len(no_config_teachers) > 0:
            logger.warning('teachers {} has no configs. \
                    This is a waste of CUDA memory'.format(','.join(no_config_teachers)))
        assert len(extra_config_teachers) == 0, "The configs of {} is invalid \
               because you do not define these teacher models".format(','.join(extra_config_teachers))
        return mimic_configs, teacher_configs

    def load_teacher_weight(self, teacher_name, teacher_model, teacher_weight_path, teacher_bn_mode):
        if teacher_weight_path:
            logger.info(f'loading {teacher_name} weight: {teacher_weight_path}')
            state = Saver.load_checkpoint(teacher_weight_path)
            if 'ema' in state:
                if "ema_state_dict" in state['ema']:
                    logger.info("Load ema pretrain model")
                    st = state['ema']['ema_state_dict']
                else:
                    st = state['model']
            else:
                st = state['model']
            teacher_model.load(st)
        if teacher_bn_mode == 'train':
            teacher_model.train()
        else:
            teacher_model.eval()

    def build_mimic(self):
        self.mimic_jobs = {}
        for key in self.mimic_configs.keys():
            teacher_models = [self.teacher_models[tmk] for tmk in self.mimic_configs[key]['t_names']]
            teacher_mimic_names = [self.teacher_configs[tmk]['mimic_name'] for tmk in self.mimic_configs[key]['t_names']]  # noqa
            loss_weight = self.mimic_configs[key].get('loss_weight', 1.0)
            warm_up_iters = self.mimic_configs[key].get('warm_up_iters', -1)
            cfgs = self.mimic_configs[key].get('cfgs', {})
            if DIST_BACKEND.backend == 'dist' and env.world_size != 1:
                self.model = self.model.module
                for i in range(len(teacher_models)):
                    teacher_models[i] = teacher_models[i].module
            self.mimic_jobs[key] = Base_Mimicker(teacher_model=teacher_models, student_model=self.model,
                                                 teacher_names=self.mimic_configs[key]['t_names'],
                                                 teacher_mimic_names=teacher_mimic_names,
                                                 student_mimic_names=self.mimic_configs[key]['student']['mimic_name'],
                                                 loss_weight=loss_weight, warm_up_iters=warm_up_iters,
                                                 configs=cfgs)

    def adjust_input(self, input, curr_subnet_num, sample_mode=None):
        # when one subnet is finetuned, the interpolation is not needed
        # or if self.image_size_list only has one choice
        if self.subnet is not None or len(self.image_size_list) == 1:
            return input

        # share_input means the same input size is shared through all subnet
        # only when the curr_subnet_num == 0, the image size is random sampled
        # the remained subnet is used the same size
        if self.share_input and curr_subnet_num == 0:
            image_size = self.sample_image_size()
        elif self.share_input and curr_subnet_num > 0:
            return input
        # when share_input is False, use the sample_mode to choose image size
        else:
            image_size = self.sample_image_size(sample_mode=sample_mode)

        # if the size is already the same with the target size, the interpolate will skip
        input = F.interpolate(input,
                              size=(image_size[2], image_size[3]),
                              mode=self.interpolation_type,
                              align_corners=False)
        return input

    def sample_image_size(self, image_size=None, sample_mode=None):
        """
        Args:
            image_size(int or list or tuple): if not None, return the 4 dimension shape
            sample_mode(['min', 'max', 'random', None]): input resolution sample mode,
                                                        sample_mode is 'random' in default
        Returns:
            image_size(tuple): 4 dimension input size
        """
        if image_size is not None:
            input_size = get_image_size_with_shape(image_size)
            return input_size
        if sample_mode is None:
            sample_mode = 'random'
        if sample_mode == 'max':
            image_size = self.image_size_list[-1]
        elif sample_mode == 'min':
            image_size = self.image_size_list[0]
        elif sample_mode == 'random' or sample_mode == 'middle':
            image_size = random.choice(self.image_size_list)
        else:
            raise ValueError('only min max random are supported')

        input_size = get_image_size_with_shape(image_size)
        return input_size

    def adjust_model(self, curr_step, curr_subnet_num, sample_mode=None):
        # calculate current sample mode
        if sample_mode is None:
            if self.sample_subnet_num > 1:
                sample_mode = self.sample_strategy[curr_subnet_num %
                                                   len(self.sample_strategy)]
            else:
                sample_mode = self.sample_strategy[curr_step %
                                                   len(self.sample_strategy)]

        # adjust model
        if self.subnet is not None:
            subnet_settings = self.sample_subnet_settings(
                sample_mode='subnet',
                subnet_settings=self.subnet.subnet_settings)
        else:
            if sample_mode in self.subnet_table.keys():
                subnet_settings = self.subnet_table[sample_mode]['subnet_settings']
                subnet_settings = self.sample_subnet_settings(
                    sample_mode='subnet',
                    subnet_settings=subnet_settings)
            else:
                subnet_settings = self.sample_subnet_settings(
                    sample_mode=sample_mode)
        return subnet_settings, sample_mode

    def sample_subnet_settings(self,
                               sample_mode='random',
                               subnet_settings=None):
        curr_subnet_settings = {}
        for name, m in self.model.named_modules():
            if not isinstance(m, BignasSearchSpace):
                continue
            if subnet_settings is None:
                if name == 'roi_head.box_subnet':
                    curr_subnet_settings[name] = curr_subnet_settings[
                        'roi_head.cls_subnet']
                    m.sample_active_subnet(
                        sample_mode='subnet',
                        subnet_settings=curr_subnet_settings[name])
                else:
                    _subnet_settings = m.sample_active_subnet(
                        sample_mode=sample_mode)
                    curr_subnet_settings[name] = _subnet_settings
            else:
                m.sample_active_subnet(sample_mode='subnet',
                                       subnet_settings=subnet_settings[name])
                curr_subnet_settings[name] = subnet_settings[name]
        return curr_subnet_settings

    def adjust_teacher(self, input, curr_subnet_num):
        if not self.config.get('distiller', False):
            return
        for key in self.mimic_configs.keys():
            self.mimic_jobs[key].prepare()

        with torch.no_grad():
            for key in self.mimic_configs.keys():
                for num, tmk in enumerate(self.mimic_configs[key]['t_names']):
                    if self.config.distiller[tmk] and curr_subnet_num == 0:
                        self.teacher_models[tmk](input)

    def get_distiller_loss(self, sample_mode, output, curr_subnet_num):
        mimic_loss = 0
        if not self.config.get('distiller', False):
            return 0

        for key in self.mimic_configs.keys():
            teacher_output = []
            for num, tmk in enumerate(self.mimic_configs[key]['t_names']):
                if not (self.config['distiller'][tmk]) and sample_mode == 'max':
                    self.teacher_output_dict[key][tmk] = self.mimic_jobs[key].t_output_maps[num]
                    for layers_mimic in self.teacher_output_dict[key][tmk].keys():
                        temp = self.teacher_output_dict[key][tmk][layers_mimic].detach()
                        self.teacher_output_dict[key][tmk][layers_mimic] = temp
                elif (not (self.config['distiller'][tmk]) and sample_mode != 'max') or \
                     (self.config.distiller[tmk] and curr_subnet_num != 0):
                    self.mimic_jobs[key].t_output_maps[num] = self.teacher_output_dict[key][tmk]
                elif self.config.distiller[tmk] and curr_subnet_num == 0:
                    self.teacher_output_dict[key][tmk] = self.mimic_jobs[key].t_output_maps[num]
                teacher_output.append(self.mimic_jobs[key].t_output_maps[num])

            if sample_mode != 'max' :
                mimic_output = self.mimic_jobs[key].mimic(s_output=output, t_output=teacher_output, job_name=key,
                                                          teacher_names=self.mimic_configs[key]['t_names'])
                mimic_loss += sum([val for name, val in mimic_output.items() if name.find('loss') >= 0])
        return mimic_loss

    def subnet_log(self, curr_subnet_num, input, subnet_settings):
        if curr_subnet_num == 0:
            self.subnet_str = ''
        self.subnet_str += '%d: ' % curr_subnet_num + 'CHW_%s_%s_%s || ' % (
            input.size(1), input.size(2), input.size(3))
        for name, settings in subnet_settings.items():
            self.subnet_str += name + ': ' + ','.join(
                ['%s_%s' % (key, '%s' % val)
                 for key, val in settings.items()]) + ' || '
        if self.config.get('distiller', False):
            for mimic_job in self.config['distiller']['mimic'].keys():
                loss_type = ''
                loss_weight = self.config['distiller']['mimic'][mimic_job]['loss_weight']
                if loss_weight > 0:
                    loss_type = '%s_%.1f_%s' % (mimic_job,
                                                loss_weight,
                                                self.distiller_type) + ' || '
                    self.subnet_str += loss_type

    def show_subnet_log(self):
        logger.info(self.subnet_str)

    def reset_subnet_running_statistics(self, model, data_loader, model_dtype):
        bn_mean = {}
        bn_var = {}
        forward_model = copy.deepcopy(model)
        forward_model.cuda()
        forward_model = DistModule(forward_model, True)
        reset_model_bn_forward(forward_model, bn_mean, bn_var)

        logger.info('calculate bn')
        iterator = iter(data_loader)
        max_iter = data_loader.get_epoch_size()
        logger.info('total iter {}'.format(max_iter))

        with torch.no_grad():
            for count in range(max_iter):
                batch = next(iterator)
                if batch['image'].device != torch.device('cuda') or batch['image'].dtype != model_dtype:
                    batch = to_device(batch, device=torch.device('cuda'), dtype=model_dtype)

                forward_model(batch)

        copy_bn_statistics(model, bn_mean, bn_var)
        logger.info('bn complete')
        return model

    def get_subnet_flops(self, image_size=None, subnet_settings=None):
        if image_size is None:
            image_size = self.sample_image_size()
        else:
            image_size = get_image_size_with_shape(image_size)

        curr_subnet_settings = self.sample_subnet_settings(
            sample_mode='random', subnet_settings=subnet_settings)

        for name, module in self.model.named_children():
            logger.info('use first part of model {} to get fake input'.format(name))
            if hasattr(module, "get_fake_input"):
                input = module.get_fake_input(input_shape=image_size)
            else:
                input = self.get_fake_input(module, image_size)
            break
        flops_dict, params_dict = count_dynamic_flops_and_params(self.model, input, depth=0)

        logger.info(
            'Subnet with settings: {}\timage_size {}\tflops {}\tparams {}'.
            format(curr_subnet_settings, image_size, clever_format(flops_dict),
                   clever_format(params_dict)))
        return flops_dict, params_dict, image_size, curr_subnet_settings

    def get_fake_input(self, module, image_size):
        input = torch.randn(image_size)
        device = next(module.parameters(), torch.tensor([])).device
        input = input.to(device)
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d) and m.weight.dtype == torch.float16:
                input = input.half()
        b, c, height, width = map(int, input.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, False]],
            'image': input,
            'filename': ['Test.jpg'],
            'label': torch.LongTensor([[0]]),
        }
        return input

    def check_flops_range(self, flops):
        self.baseline_flops = parse_flops(self.subnet.get('baseline_flops', None))
        self.large_flops = self.subnet.get('large_flops', False)

        v1 = flops['total']
        if v1 <= min(self.flops_range) or v1 >= max(self.flops_range):
            logger.info('current flops {} do not match target flops {}'.format(
                clever_format(v1), clever_format(self.flops_range)))
            return False

        if self.baseline_flops and self.large_flops:
            for (k1, v1), (k2, v2) in zip(flops.items(),
                                          self.baseline_flops.items()):
                assert k1 == k2
                if v1 < v2 * 0.9:
                    logger.info(
                        'current flops {} do not match target flops {} in {}'.format(
                            clever_format(v1), clever_format(v2), k1))
                    return False
        return True

    def get_subnet_num(self,
                       count_flops=False,
                       save_subnet=False,
                       print_log=False):
        total_count = 1
        module_config = {}
        for name, m in self.model.named_modules():
            if not isinstance(m, BignasSearchSpace):
                continue
            kwargs_configs = get_kwargs_itertools(m.dynamic_settings)
            count = 0
            module_range = []
            for kwargs in itertools.product(*kwargs_configs.values()):
                kwarg_config = dict(zip(kwargs_configs.keys(), kwargs))
                module_range.append(kwarg_config)
                if print_log:
                    logger.info(json.dumps(kwarg_config))
                count += 1
                if count > 5000000:
                    logger.info(
                        'total subnet number for {} surpass 5000000'.format(
                            name))
                    break
            logger.info('total subnet number for {} is {}'.format(
                name, count))
            total_count = total_count * count
            module_config[name] = module_range

        logger.info('all subnet number is {}'.format(total_count))
        subnet_table = {}
        count = 0
        for kwargs in itertools.product(*module_config.values()):
            kwarg_config = dict(zip(module_config.keys(), kwargs))
            subnet_table[count] = {'image_size': self.config.data.image_size_list[0], 'subnet_settings': kwarg_config}
            if count_flops:
                flops, params, image_size, subnet_settings = self.get_subnet_flops(
                    image_size=subnet_table[count]['image_size'],
                    subnet_settings=subnet_table[count]['subnet_settings'])
                subnet_table[count]['flops'] = flops
                subnet_table[count]['params'] = params
            count += 1
        return subnet_table

    def sample_subnet_lut(self, test_latency=False):
        """
        Args:
            test_latency (bool): if test latency or not
        Returns:
            subnet_table (dict): a dict contains subnet_settings and flops and latency
        """
        # process settings
        assert self.subnet is not None
        self.lut_path = self.subnet.get('lut_path', None)
        self.flops_range = parse_flops(self.subnet.get('flops_range', None))
        self.subnet_count = self.subnet.get('subnet_count', 500)
        self.subnet_sample_mode = self.subnet.get('subnet_sample_mode',
                                                  'random')
        rewrite_batch_norm_bias(self.model)

        if self.flops_range is not None:
            logger.info('flops range with {}'.format(clever_format(self.flops_range)))
        else:
            logger.info('No flops range defined')

        # get subnet table
        if self.subnet_sample_mode == 'traverse':
            subnet_table = self.get_subnet_num(count_flops=True, save_subnet=True)
        else:
            subnet_table = {}
            logger.info('subnet count {}'.format(self.subnet_count))
            count = 0
            seed = torch.Tensor([int(time.time() * 10000) % 10000])
            logger.info('seed {}'.format(seed))
            broadcast(seed, root=0)
            while count < self.subnet_count:
                seed += 1
                broadcast(seed, root=0)
                random.seed(seed.item())
                flops, params, image_size, subnet_settings = self.get_subnet_flops()
                if self.flops_range is not None:
                    if not self.check_flops_range(flops):
                        logger.info('do not match target fp flops')
                        continue
                subnet_table[count] = {
                    'flops': flops,
                    'params': params,
                    'image_size': image_size,
                    'subnet_settings': subnet_settings
                }
                logger.info('current subnet count {} with {}'.format(
                    count, clever_dump_table(subnet_table[count], ['flops', 'params'])))
                count += 1

        # test latency for every subnet in the table
        if self.latency is not None and test_latency:
            raise NotImplementedError

        subnet_table_float = copy.deepcopy(subnet_table)
        logger.info('--------------subnet table--------------')
        logger.info(clever_dump_table(subnet_table, ['flops', 'params']))
        return subnet_table, subnet_table_float

    def get_subnet_weight(self, subnet_settings=None):
        new_settings = adapt_settings(subnet_settings)

        static_model = copy.deepcopy(self.model)
        inplanes = None
        for name, m in self.model.named_children():
            if hasattr(m, 'sample_active_subnet_weights'):
                # it is used to extract blocks
                if hasattr(m, 'inplanes'):
                    m.inplanes = inplanes
                subnet = m.sample_active_subnet_weights(new_settings[name], inplanes)
                setattr(static_model, name, subnet)
                if hasattr(subnet, "get_outplanes"):
                    inplanes = subnet.get_outplanes()
            else:
                setattr(static_model, name, m)
        return static_model
