import os
import cv2
import numpy as np
import json
import copy
import torch
import yaml
from easydict import EasyDict

from up.utils.general.tocaffe_helper import to_caffe
from up.utils.general.cfg_helper import merge_opts_into_cfg
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import TOKESTREL_REGISTRY, KS_PROCESSOR_REGISTRY, KS_PARSER_REGISTRY


def generate_image_bins_cls(image_list, mean, std, resize_hw, image_bins_folder='./image_bins'):
    """
    Generate data for calibration.
    """
    if not os.path.exists(image_bins_folder):
        os.makedirs(image_bins_folder)
    else:
        os.system("rm -r {}/*".format(image_bins_folder))

    for i, image in enumerate(image_list):
        output_bin = os.path.join(image_bins_folder, str(i) + ".bin")
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, resize_hw)
        if len(mean) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img, dtype=np.float32)
            img = (img - mean[0]) / std[0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
            for i in range(len(mean)):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
            img = img.transpose((2, 0, 1))
        bin_str = img.astype('f').tostring()
        with open(output_bin, 'wb') as fp:
            fp.write(bin_str)
    return image_bins_folder


def generate_nnie_config_cls(nnie_cfg, config, nnie_out_path='./config.json', tensor_type='float'):
    """
    Generate NNIE config for spring.nart.switch, details in
    http://spring.sensetime.com/docs/nart/tutorial/switch/nnie.html
    """
    u8_start = False if tensor_type == 'float' else False
    default_config = {
        "default_net_type_token": "nnie",
        "rand_input": False,
        "data_num": 100,
        "input_path_map": {
            "data": "./image_bins",
        },
        "nnie": {
            "max_batch": 1,
            "output_names": [],
            "mapper_version": 11,
            "u8_start": u8_start,
            "device": "gpu",
            "verbose": False,
            "image_path_list": ["./image_list.txt"],
            "mean": [128, 128, 128],
            "std": [1, 1, 1]
        }
    }
    image_path_list = nnie_cfg['image_path_list']
    assert os.path.exists(image_path_list)
    with open(image_path_list, 'r') as f:
        image_list = [item.strip() for item in f.readlines()]

    mean = config['to_kestrel'].get('pixel_means', [123.675, 116.28, 103.53])
    std = config['to_kestrel'].get('pixel_stds', [58.395, 57.12, 57.375])
    resize_hw = config['to_kestrel'].get('resize_hw', (224, 224))
    resize_hw = tuple(resize_hw)
    data_num = len(image_list)
    image_bin_path = generate_image_bins_cls(image_list, mean, std, resize_hw)
    default_config['data_num'] = data_num
    default_config['input_path_map']['data'] = image_bin_path
    default_config['nnie']['max_batch'] = nnie_cfg.get('max_batch', 1)
    default_config['nnie']['mapper_version'] = nnie_cfg.get('mapper_version', 11)
    default_config['nnie']['image_path_list'] = [image_path_list]
    default_config['nnie']['mean'] = [128] * len(std)
    default_config['nnie']['std'] = [1] * len(std)
    with open(nnie_out_path, "w") as f:
        json.dump(default_config, f, indent=2)

    return nnie_out_path


def generate_nnie_config(nnie_cfg, config, nnie_out_path='./config.json'):
    default_config = {
        "default_net_type_token": "nnie",
        "rand_input": False,
        "data_num": 100,
        "input_path_map": {
            "data": "./image_bins",
        },
        "nnie": {
            "max_batch": 1,
            "output_names": [],
            "mapper_version": 11,
            "device": "gpu",
            "u8_start": True,
            "verbose": False,
            "image_path_list": ["./image_list.txt"],
            "mean": [128],
            "std": [1]
        }
    }
    image_path_list = nnie_cfg['image_path_list']
    assert os.path.exists(image_path_list)
    with open(image_path_list, "r") as f:
        image_list = [item.strip() for item in f.readlines()]
    mean = [i * 255 for i in config['normalize']['kwargs']['mean']]
    std = [i * 255 for i in config['normalize']['kwargs']['std']]
    resize_hw = config['to_kestrel'].get('resize_hw', "")
    data_num = len(image_list)
    image_bin_path = generate_image_bins(image_list, mean, std, resize_hw)
    default_config['data_num'] = data_num
    default_config["input_path_map"]["data"] = image_bin_path
    default_config['nnie']["max_batch"] = nnie_cfg.get('max_batch', 1)
    default_config['nnie']['mapper_version'] = nnie_cfg.get('mapper_version', 11)
    default_config['nnie']['image_path_list'] = [image_path_list]
    default_config['nnie']['mean'] = mean
    default_config['nnie']['std'] = std
    # hard code for rgb nnie convert
    if len(mean) == 3:
        default_config['nnie'].pop('mean')
        default_config['nnie'].pop('std')
        default_config['nnie'].pop('image_path_list')
        default_config['nnie']['u8_start'] = False
    with open(nnie_out_path, "w") as f:
        json.dump(default_config, f, indent=2)
    return nnie_out_path


def generate_image_bins(image_list, mean, std, resize_hw="", image_bins_folder='./image_bins'):
    if not os.path.exists(image_bins_folder):
        os.makedirs(image_bins_folder)
    else:
        os.system("rm -r {}/*".format(image_bins_folder))
    for i, image in enumerate(image_list):
        output_bin = os.path.join(image_bins_folder, str(i) + ".bin")
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        assert resize_hw != '', "resize_hw must be provided"
        h, w = [int(i) for i in resize_hw.strip().split("x")]
        img = cv2.resize(img, (h, w))
        if len(mean) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img, dtype=np.float32)
            img = (img - mean[0]) / std[0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
            for i in range(len(mean)):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
            img = img.transpose((2, 0, 1))
        bin_str = img.astype('f').tostring()
        with open(output_bin, 'wb') as fp:
            fp.write(bin_str)
    return image_bins_folder


@TOKESTREL_REGISTRY.register('pod')
class ToKestrel(object):
    def __init__(self, config, caffemodel, parameters, save_to=None, serialize=False, input_channel=3):
        self.config = config
        self.caffemodel = caffemodel
        self.parameters = parameters
        self.save_to = save_to
        self.serialize = serialize
        self.input_channel = input_channel

    def process(self):
        cmd = 'python -m spring.nart.tools.caffe.convert -a {}'.format(self.caffemodel)
        os.system(cmd)
        logger.info('=========merge bn done==========')

        prefix = self.caffemodel.rsplit('.', 1)[0] + '-convert'
        caffemodel = '{}.caffemodel'.format(prefix)
        prototxt = '{}.prototxt'.format(prefix)

        # we get anchors.json, $prefix.prototxt, $prefix.caffemodel
        config = copy.deepcopy(self.config)
        plugin = config['to_kestrel']['plugin']
        detector = config['to_kestrel']['detector']
        nnie_cfg = config['to_kestrel'].get('nnie', None)

        if config['to_kestrel'].get('sigmoid', None) is not None:
            self.parameters['sigmoid'] = True
        if not config['to_kestrel'].get('retina_share_location', True):
            self.parameters['retina_share_location'] = False
        if config['to_kestrel'].get('retina_class_first', False):
            self.parameters['retina_class_first'] = True

        version = config['to_kestrel'].get('version', "1.0.0")
        # kestrel_model = '{}_{}.tar'.format(detector, version)
        parameters_json = 'tmp_parameters_json'
        # convert torch.tensor in parameters to int or list
        for key in self.parameters:
            if torch.is_tensor(self.parameters[key]):
                self.parameters[key] = self.parameters[key].tolist()
        with open(parameters_json, 'w') as f:
            json.dump(self.parameters, f, indent=2)
        resize_hw = config['to_kestrel'].get('resize_hw', '')
        nnie = False
        if nnie_cfg is not None and not self.serialize:
            logger.info("auto nnie ")
            nnie = True
            nnie_cfg_path = generate_nnie_config(nnie_cfg, config)
            nnie_cmd = 'python -m spring.nart.switch  -c {} -t nnie {} {}'.format(nnie_cfg_path, prototxt, caffemodel)
            os.system(nnie_cmd)

        if self.save_to is None:
            self.save_to = config['to_kestrel']['save_to']

        ks_processor = KS_PROCESSOR_REGISTRY[plugin](prototxt,
                                                     caffemodel, b=1,
                                                     n=detector, v=version,
                                                     p=self.save_to,
                                                     k=parameters_json,
                                                     i=self.input_channel,
                                                     s=self.serialize,
                                                     resize_hw=resize_hw,
                                                     nnie=nnie)
        ks_processor.process()
        # shutil.move(kestrel_model, self.save_to)
        logger.info('save kestrel model to: {}'.format(self.save_to))
        return self.save_to


@TOKESTREL_REGISTRY.register('cls')
class CLSToKestrel(object):
    def __init__(self, config, caffemodel, parameters=None, save_to=None, serialize=False, input_channel=3):
        self.config = config
        self.caffemodel = caffemodel
        self.save_to = save_to

    def refactor_config(self):
        kestrel_config = EasyDict()
        kestrel_config['pixel_means'] = self.config['to_kestrel'].get('pixel_means', [123.675, 116.28, 103.53])
        kestrel_config['pixel_stds'] = self.config['to_kestrel'].get('pixel_stds', [58.395, 57.12, 57.375])

        kestrel_config['is_rgb'] = self.config['to_kestrel'].get('is_rgb', True)
        kestrel_config['save_all_label'] = self.config['to_kestrel'].get('save_all_label', True)
        kestrel_config['type'] = self.config['to_kestrel'].get('type', 'ImageNet')

        if self.config.get('to_kestrel') and self.config['to_kestrel'].get('class_label'):
            kestrel_config['class_label'] = self.config['to_kestrel']['class_label']
        else:
            kestrel_config['class_label'] = {}
            kestrel_config['class_label']['imagenet'] = {}
            kestrel_config['class_label']['imagenet']['calculator'] = 'bypass'
            num_classes = self.config.get('num_classes', 1000)
            kestrel_config['class_label']['imagenet']['labels'] = [str(i) for i in np.arange(num_classes)]
            kestrel_config['class_label']['imagenet']['feature_start'] = 0
            kestrel_config['class_label']['imagenet']['feature_end'] = num_classes - 1

        self.kestrel_config = kestrel_config

    def to_nnie(self, nnie_cfg, config, prototxt, caffemodel, model_name):
        nnie_cfg_path = generate_nnie_config(nnie_cfg, config)
        nnie_cmd = 'python -m spring.nart.switch -c {} -t nnie {} {}'.format(nnie_cfg_path, prototxt, caffemodel)

        os.system(nnie_cmd)
        assert os.path.exists('parameters.json')
        with open('parameters.json', 'r') as f:
            params = json.load(f)
        params['model_files']['net']['net'] = 'engine.bin'
        params['model_files']['net']['backend'] = 'kestrel_nart'
        with open('parameters.json', 'w') as f:
            json.dump(params, f, indent=2)
        tar_cmd = 'tar cvf {} engine.bin engine.bin.json meta.json meta.conf parameters.json category_param.json' \
            .format(model_name, + '_nnie.tar')
        os.system(tar_cmd)
        logger.info("generate {model_name + '_nnie.tar'} done!")

    def process(self):
        prefix = self.caffemodel.rsplit('.', 1)[0]
        caffemodel = '{}.caffemodel'.format(prefix)
        prototxt = '{}.prototxt'.format(prefix)
        version = self.config['to_kestrel'].get('version', '1.0.0')
        model_name = self.config['to_kestrel'].get('model_name', 'kestrel_model')

        kestrel_model = '{}_{}.tar'.format(model_name, version)
        to_kestrel_yml = 'temp_to_kestrel.yml'
        self.refactor_config()

        with open(to_kestrel_yml, 'w') as f:
            yaml.dump(json.loads(json.dumps(self.kestrel_config)), f)

        cmd = 'python -m spring.nart.tools.kestrel.classifier {} {} -v {} -c {} -n {} -p {}'.format(
            prototxt, self.caffemodel, version, to_kestrel_yml, model_name, self.save_to)
        mv_cmd = 'mv model* temp_to_kestrel.yml -t ./to_kestrel/'

        logger.info('Converting Model to Kestrel...')
        os.system(cmd)
        os.system(mv_cmd)
        logger.info('To Kestrel Done!')

        if self.save_to is None:
            self.save_to = kestrel_model
        else:
            self.save_to = os.path.join(self.save_to, kestrel_model)
        logger.info('Save kestrel model to: {}'.format(self.save_to))

        # convert model to nnie
        nnie_cfg = self.config['to_kestrel'].get('nnie', None)
        if nnie_cfg is not None:
            logger.info('Converting Model to NNIE...')
            self.to_nnie(nnie_cfg, self.config, prototxt, caffemodel, model_name)

        return self.save_to


def get_kestrel_parameters(config):
    plugin = config['to_kestrel']['plugin']
    parser = KS_PARSER_REGISTRY[plugin](config)
    return parser.get_kestrel_parameters()


def to_kestrel(config, save_to=None, serialize=False):
    opts = config.get('args', {}).get('opts', [])
    config = merge_opts_into_cfg(opts, config)
    input_channel = config['to_kestrel'].get('input_channel', 3)
    resize_hw = None
    if config['to_kestrel'].get('resize_hw', '') != '':
        resize_hw = config['to_kestrel'].get('resize_hw', '640x1024')
        resize_hw = (int(i) for i in resize_hw.strip().split("x"))
        resize_hw = (input_channel, *resize_hw)
    caffemodel_name = to_caffe(config, input_size=resize_hw, input_channel=input_channel)
    toks_type = config['to_kestrel'].get('toks_type', 'pod')
    parameters = None
    if toks_type == 'pod':
        parameters = get_kestrel_parameters(config)
        logger.info(f'parameters:{parameters}')
    tokestrel_ins = TOKESTREL_REGISTRY[toks_type](config,
                                                  caffemodel_name,
                                                  parameters,
                                                  save_to,
                                                  serialize,
                                                  input_channel)
    save_to = tokestrel_ins.process()

    return save_to
