from __future__ import division

import sys
import cv2
import torch
import numpy as np
from up.apis.inference import BaseInference
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader

from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY


@SUBCOMMAND_REGISTRY.register('inference_video')
class Inference_video(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name, help='sub-command for inference')
        sub_parser.add_argument('--config', type=str, required=True, help='path to yaml config')
        sub_parser.add_argument('--ckpt', type=str, required=True, help='path of model')
        sub_parser.add_argument('--work_dir', type=str, default='./', help='path to work directory')
        sub_parser.add_argument('--video_file', type=str, default='', help='path to video')
        sub_parser.add_argument('--output_file', type=str, default='./out.mp4', help='output video file path')
        sub_parser.add_argument('--vis_dir', type=str, default='./', help='path to visualize directory')
        sub_parser.add_argument('--score_thresh', type=float, default=0.5, help='bbox score thresh to visualize')

        sub_parser.set_defaults(run=_main)

        return sub_parser


def _main(args):
    video_file = args.video_file
    logger.info("video file: {}".format(video_file))
    if (video_file == ''):
        logger.info("video file LACKS!")
        sys.exit()

    cfg = load_yaml(args.config, 'up')
    cfg['args'] = {
        'video_file': args.video_file,
        'ckpt': args.ckpt,
        'output_file': args.output_file,
        'image_path': args.video_file,
        'vis_dir': args.vis_dir,
    }

    infer_cfg = cfg.get('inferencer', {})
    if len(infer_cfg) == 0:
        infer_cfg = {'visualizer': {'type': 'plt', 'kwargs': {'thresh': args.score_thresh}}}

    if infer_cfg['visualizer']['kwargs'].get('ext', None) is None:
        infer_cfg['visualizer']['kwargs'].setdefault('ext', 'jpg')
    else:
        infer_cfg['visualizer']['kwargs']['ext'] = 'jpg'

    if infer_cfg['visualizer']['kwargs'].get('vis_dir', None) is None:
        infer_cfg['visualizer']['kwargs'].setdefault('vis_dir', args.vis_dir)

    inferencor = BaseInference(cfg, **infer_cfg)

    cap = cv2.VideoCapture(video_file)
    frames_num = int(cap.get(7))
    wd = int(cap.get(3))
    hg = int(cap.get(4))
    # video writer
    out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (wd, hg))

    for img_idx in range(0, frames_num):
        logger.info("cur idx: {}".format(str(img_idx)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, img_idx)
        ret, frame = cap.read()
        if ret is False:
            logger.info("error occurs (false)!")
            sys.exit()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.info('processing {}:{}'.format(img_idx, video_file))
        batch = inferencor.fetch([frame])
        with torch.no_grad():
            output = inferencor.detector(batch)
        output = inferencor.map_back(output)[0]
        # output = predictor.predict([frame], [img_idx])[0]
        img = output['image']
        bboxes = output['dt_bboxes']

        classes = bboxes[:, -1].astype(np.int32)
        boxes = bboxes[:, :-1]

        filename = "{}".format(img_idx)
        inferencor.visualizer.vis(img, boxes, classes, filename)
        img = cv2.imread(args.vis_dir + filename + '.jpg')
        out.write(img)

    out.release()
