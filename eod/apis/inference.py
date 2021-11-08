import os
import cv2
import numpy as np
import torch
from torch.nn.modules.utils import _pair
from easydict import EasyDict

from eod.data.datasets.transforms import build_transformer
from eod.data.data_utils import get_image_size
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY, BATCHING_REGISTRY, IMAGE_READER_REGISTRY
from eod.utils.general.registry_factory import INFERENCER_REGISTRY, SAVER_REGISTRY, VISUALIZER_REGISTRY

__all__ = ['BaseInference']


@INFERENCER_REGISTRY.register('base')
class BaseInference(object):
    def __init__(self, config, work_dir='./'):
        self.args = config.get('args', {})
        self.config = config
        # cfg_infer = config['inference']
        self.class_names = config.get('class_names', None)
        self.work_dir = work_dir
        self.ckpt = self.args['ckpt']
        self.vis_dir = self.args['vis_dir']
        self.image_path = self.args['image_path']

        assert self.image_path and os.path.exists(self.image_path), 'Invalid images path.'

        # build DataFetch
        self.build_data()
        logger.info('build data fetcher done')
        # build model
        self.build_model()
        logger.info('build model done')
        # build saver
        self.build_saver()
        logger.info('build saver done')
        # build visualizer
        self.vis_type = config['inference']['visualizer']['type']
        # update vis_dir
        config['inference']['visualizer']['kwargs']['vis_dir'] = self.vis_dir
        self.visualizer = VISUALIZER_REGISTRY.build(config['inference']['visualizer'])
        logger.info('build visualizer done')
        # resume
        self.resume()
        logger.info('load weights done')

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x

    def resume(self):
        checkpoint = self.saver.load_checkpoint(self.ckpt)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', {}))
        self.detector.load(state_dict, strict=False)

    def build_saver(self):
        cfg_saver = self.config['saver']
        _cfg_saver = cfg_saver
        if 'kwargs' not in cfg_saver:
            _cfg_saver = {"type": 'base', "kwargs": {}}
            _cfg_saver['kwargs']['save_cfg'] = cfg_saver
            _cfg_saver['kwargs']['work_dir'] = self.work_dir
        self.saver = SAVER_REGISTRY.build(_cfg_saver)

    def build_model(self):
        model_helper_cfg = self.config['runtime'].get('model_helper', {})
        model_helper_cfg['type'] = model_helper_cfg.get('type', 'base')
        model_helper_cfg['kwargs'] = model_helper_cfg.get('kwargs', {'cfg': self.config['net']})
        self.detector = MODEL_HELPER_REGISTRY.build(model_helper_cfg).cuda().eval()

    def build_data(self):
        data_cfg = self.config['dataset']
        assert 'test' in data_cfg, 'Test dataset config must need !'
        dataset_cfg = data_cfg['test']['dataset']['kwargs']
        self.color_mode = dataset_cfg['image_reader']['kwargs']['color_mode']
        # build image_reader
        self.image_reader = IMAGE_READER_REGISTRY.build(dataset_cfg['image_reader'])

        self.transformer = build_transformer(dataset_cfg['transformer'])
        pad_type = data_cfg['dataloader']['kwargs'].get('pad_type', 'batch_pad')
        pad_value = data_cfg['dataloader']['kwargs'].get('pad_value', 0)
        alignment = data_cfg['dataloader']['kwargs']['alignment']
        self.batch_pad = BATCHING_REGISTRY.get(pad_type)(alignment, pad_value)

    def iterate_image(self, image_dir):
        EXTS = ['jpg', 'jpeg', 'png', 'svg', 'bmp']

        for root, subdirs, subfiles in os.walk(image_dir):
            for filename in subfiles:
                ext = filename.rsplit('.', 1)[-1].lower()
                filepath = os.path.join(root, filename)
                if ext in EXTS:
                    yield filepath

    def map_back(self, output):
        """Map predictions to original image
        Args:
           - output: dict
        Returns:
           - output_list: list of dict,
        """
        origin_images = output['origin_image']
        image_info = output['image_info']
        bboxes = self.tensor2numpy(output['dt_bboxes'])
        batch_size = len(image_info)

        output_list = []
        for b_ix in range(batch_size):

            origin_image = origin_images[b_ix]
            if origin_image.ndim == 3:
                origin_image_h, origin_image_w, _ = origin_image.shape
            else:
                origin_image_h, origin_image_w = origin_image.shape

            img_info = image_info[b_ix]
            unpad_image_h, unpad_image_w = img_info[:2]
            scale_h, scale_w = _pair(img_info[2])
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]

            # resize bbox
            img_bboxes = bboxes[keep_ix]
            img_bboxes[:, 1] /= scale_w
            img_bboxes[:, 2] /= scale_h
            img_bboxes[:, 3] /= scale_w
            img_bboxes[:, 4] /= scale_h
            img_bboxes = img_bboxes[:, 1:]

            img_output = {
                'image': origin_image,
                'image_info': img_info,
                'dt_bboxes': img_bboxes
            }
            output_list.append(img_output)

        return output_list

    def fetch_single(self, filename):
        img = self.image_reader.read(filename)
        data = EasyDict({
            'filename': filename,
            'origin_image': img,
            'image': img,
            'flipped': False
        })
        data = self.transformer(data)
        scale_factor = data.get('scale_factor', 1)

        image_h, image_w = get_image_size(img)
        new_image_h, new_image_w = get_image_size(data.image)
        pad_w, pad_h = data.get('dw', 0), data.get('dh', 0)
        data.image_info = [new_image_h, new_image_w, scale_factor, image_h, image_w, data.flipped, pad_w, pad_h, filename]
        data.image = data.image.cuda()
        return data

    def fetch(self, filename_list):
        batch = [self.fetch_single(filename) for filename in filename_list]

        batch_keys = list(batch[0].keys())

        def batch_value(key, default=None):
            return [_.get(key, default) for _ in batch]

        data = EasyDict({k: batch_value(k) for k in batch_keys})
        data = self.batch_pad(data)

        return data

    def predict(self):
        output_list = []
        if os.path.isdir(self.image_path):
            list_imgs = self.iterate_image(self.image_path)
        else:
            list_imgs = [self.image_path]
        for img_idx, filename in enumerate(list_imgs):
            logger.info('predicting {}:{}'.format(img_idx, filename))
            batch = self.fetch([filename])
            with torch.no_grad():
                output = self.detector(batch)
            output = self.map_back(output)
            self.vis(output)

    def vis(self, outputs):
        for img_idx, output in enumerate(outputs):
            img = output['image']
            if self.color_mode != 'RGB':
                cvt_color_vis = getattr(cv2, 'COLOR_{}2RGB'.format(self.color_mode))
                img = cv2.cvtColor(img, cvt_color_vis)
            boxes = output['dt_bboxes']
            filename = os.path.basename(output['image_info'][-1])
            if self.vis_type == 'plt':
                filename = filename.rsplit('.', 1)[0]
            logger.info('visualizing {}'.format(filename))

            img_h, img_w = img.shape[:2]
            classes = boxes[:, -1].astype(np.int32)
            boxes = boxes[:, :-1]
            self.visualizer.vis(img, boxes, classes, filename)
