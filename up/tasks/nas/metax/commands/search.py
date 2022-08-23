from __future__ import division

# Standard Library
import argparse
import sys
import json

# Import from third library
import torch.nn as nn
import torch.multiprocessing as mp

from up.utils.env.dist_helper import setup_distributed, finalize, env
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.env.launch import launch
from up.utils.general.user_analysis_helper import send_info
from up.utils.general.log_helper import default_logger as logger
from up.commands.subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from up.utils.general.global_flag import DIST_BACKEND
from up.tasks.nas.metax.models.backbones import xmnet_search

try:
    from metax.actor import RestfulActor
except:  # noqa
    RestfulActor = None


__all__ = ['Search']


class ClsWrapper(nn.Module):
    def __init__(self, detector, add_softmax=False):
        super(ClsWrapper, self).__init__()
        self.detector = detector
        self.add_softmax = add_softmax
        if self.add_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        b, c, height, width = map(int, image.size())
        input = {
            'image_info': [[height, width, 1.0, height, width, 0]] * b,
            'image': image
        }
        print(f'before detector forward')
        output = self.detector(input)
        return output['features']


@SUBCOMMAND_REGISTRY.register('search')
class Search(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for searching',
                                       help='searching a model')

        sub_parser.add_argument('-e',
                                '--evaluate',
                                dest='evaluate',
                                action='store_true',
                                help='evaluate model on validation set')
        sub_parser.add_argument(
            '--fork-method',
            dest='fork_method',
            type=str,
            default='fork',
            choices=['spawn', 'fork'],
            help='method to fork subprocess, especially for dataloader')
        sub_parser.add_argument('--backend',
                                dest='backend',
                                type=str,
                                default='linklink',
                                help='model backend')
        sub_parser.add_argument(
            '--nocudnn',
            dest='nocudnn',
            action='store_true',
            help='Whether to use cudnn backend or not. Please disable cudnn when running on V100'
        )
        sub_parser.add_argument(
            '--allow_dead_parameter',
            action='store_true',
            help='dead parameter (defined in model but not used in forward pass) is allowed'
        )
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--display',
                                dest='display',
                                type=int,
                                default=20,
                                help='display intervel')
        sub_parser.add_argument('--async',
                                dest='asynchronize',
                                action='store_true',
                                help='whether to use asynchronize mode(linklink)')
        sub_parser.add_argument('--ng', '--num_gpus_per_machine',
                                dest='num_gpus_per_machine',
                                type=int,
                                default=8,
                                help='num_gpus_per_machine')
        sub_parser.add_argument('--nm', '--num_machines',
                                dest='num_machines',
                                type=int,
                                default=1,
                                help='num_machines')
        sub_parser.add_argument('--launch',
                                dest='launch',
                                type=str,
                                default='slurm',
                                help='launch backend')
        sub_parser.add_argument('--port',
                                dest='port',
                                type=int,
                                default=13333,
                                help='dist port')
        sub_parser.add_argument('--no_running_config',
                                action='store_true',
                                help='disable display running config')
        sub_parser.add_argument('--phase', default='train', help="train phase")
        sub_parser.add_argument('--cfg_type',
                                dest='cfg_type',
                                type=str,
                                default='up',
                                help='config type (up or pod)')
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)
        sub_parser.add_argument('--actor_conf',
                                help='metax actor config file',
                                type=str,
                                required=True)
        sub_parser.add_argument('--log_dir',
                                help='the path of log ',
                                type=str,
                                required=True)
        sub_parser.add_argument('--save_prefix',
                                help='the path of onnx ',
                                type=str,
                                required=True)

        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'ddp': args.backend == 'dist',
        'config_path': args.config,
        'asynchronize': args.asynchronize,
        'nocudnn': args.nocudnn,
        'display': args.display,
        'no_running_config': args.no_running_config,
        'allow_dead_parameter': args.allow_dead_parameter,
        'opts': args.opts
    }
    train_phase = args.phase
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg

    actor = RestfulActor(args.actor_conf)

    cells = actor.find_target()

    cfg['net'][0]['kwargs']['cells'] = cells

    logger.info(cfg['net'][0]['kwargs']['cells'])

    training = True
    if args.evaluate:
        training = False
        train_phase = "eval"
    send_info(cfg, train_phase)
    runner_cfg['kwargs']['training'] = training
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    train_func = {"train": runner.train, "eval": runner.evaluate}
    train_func[train_phase]()
    if env.world_size > 1:
        finalize()
    performance = 0.

    logger.info(f'========log path: {args.log_dir}')

    for line in open(args.log_dir, "r"):
        if "best val" in line:
            performance = float(line.split('val: ')[1])
            logger.info(f'========get performance: {performance}')
            break

    if env.is_master():
        # latency
        model_search = xmnet_search(cells=cells)
        model_search.eval()
        model_conv = ClsWrapper(model_search)

        import spring.nart.tools.pytorch as pytorch
        from up.utils.general.registry_factory import LATENCY_REGISTRY

        with pytorch.convert_mode():
            pytorch.convert_v2(model_conv, [(3, 224, 224)],
                               args.save_prefix,
                               use_external_data_format=False)

        cfg_gdbp = cfg.get('gdbp', None)
        assert cfg_gdbp is not None, "gdbp config should be supported."
        latency_type = cfg_gdbp.pop('latency_type', 'base')
        latency_ins = LATENCY_REGISTRY[latency_type](cfg_gdbp, args.save_prefix + '.onnx')
        latency_ins.process()

        res_json = cfg_gdbp.get('res_json', 'latency_test.json')
        f_res = open(res_json, 'r')
        ret_val = f_res.readlines()[0]
        ret_val = json.loads(ret_val.strip())
        f_res.close()

        if ret_val['ret']['status'] == 'success':
            latency = ret_val['cost_time']  # unit of return cost time is ms
            target_latency = 1.18254
            params = ret_val['params'] / 1e6
            flops = ret_val['flops'] / 1e9
            reward = performance - 0.05 * abs(latency / target_latency - 1)
            logger.info(f"latency: {latency}")
            logger.info(f"param: {params}")
            logger.info(f"flops: {flops}")
            logger.info(f"reward: {reward}")
        else:
            logger.info(f"faild")

        ret = actor.send_result(reward, acc=performance, params={'param': params, 'flops': flops, 'latency': latency})  # noqa


def _main(args):
    DIST_BACKEND.backend = args.backend
    if args.launch == 'pytorch':
        launch(main, args.num_gpus_per_machine, args.num_machines, args=args, start_method=args.fork_method)
    else:
        mp.set_start_method(args.fork_method, force=True)
        fork_method = mp.get_start_method(allow_none=True)
        assert fork_method == args.fork_method
        sys.stdout.flush()
        setup_distributed(args.port, args.launch, args.backend)
        main(args)
