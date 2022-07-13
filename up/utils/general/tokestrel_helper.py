import os
import cv2
import numpy as np
import json
import copy
import torch

from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import TOKESTREL_REGISTRY, KS_PROCESSOR_REGISTRY

__all__ = ['DetToKestrel', 'ClsToKestrel', 'SegToKestrel', 'KpToKestrel', 'Det3dToKestrel']


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


@TOKESTREL_REGISTRY.register('kp')
class KpToKestrel(object):
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

        config = copy.deepcopy(self.config)
        plugin = config['to_kestrel']['plugin']
        model_name = config['to_kestrel'].get('model_name', 'model')
        resize_hw = config['to_kestrel'].get('resize_hw', '')

        to_kestrel_yml = 'temp_to_kestrel.yml'
        version = config['to_kestrel'].get('version', "1.0.0")
        with open(to_kestrel_yml, 'w') as f:
            json.dump(self.parameters, f, indent=2)

        if self.save_to is None:
            self.save_to = config['to_kestrel']['save_to']

        ks_processor = KS_PROCESSOR_REGISTRY[plugin](prototxt,
                                                     caffemodel, b=1,
                                                     n=model_name, v=version,
                                                     p=self.save_to,
                                                     k=to_kestrel_yml,
                                                     i=self.input_channel,
                                                     s=self.serialize,
                                                     resize_hw=resize_hw)
        ks_processor.process()

        logger.info('save kestrel model to: {}'.format(self.save_to))
        return os.path.join(self.save_to, '{}_{}.tar'.format(self.save_to, version))


@TOKESTREL_REGISTRY.register('seg')
class SegToKestrel(object):
    def __init__(self, config, caffemodel, parameters, save_to=None, serialize=False, input_channel=3):
        self.config = config
        self.caffemodel = caffemodel
        self.parameters = parameters
        self.save_to = save_to
        self.serialize = serialize
        self.input_channel = input_channel

    def process(self):
        # merge bn
        cmd = 'python -m spring.nart.tools.caffe.convert -a {}'.format(self.caffemodel)
        os.system(cmd)
        logger.info('=========merge bn done==========')

        prefix = self.caffemodel.rsplit('.', 1)[0] + '-convert'
        caffemodel = '{}.caffemodel'.format(prefix)
        prototxt = '{}.prototxt'.format(prefix)

        config = copy.deepcopy(self.config)
        plugin = config['to_kestrel']['plugin']
        model_name = config['to_kestrel'].get('model_name', 'model')

        version = config['to_kestrel'].get('version', "1.0.0")
        parameters_json = 'tmp_parameters_json'
        # convert torch.tensor in parameters to int or list
        for key in self.parameters:
            if torch.is_tensor(self.parameters[key]):
                self.parameters[key] = self.parameters[key].tolist()
        with open(parameters_json, 'w') as f:
            json.dump(self.parameters, f, indent=2)
        resize_hw = config['to_kestrel'].get('resize_hw', '')

        if self.save_to is None:
            self.save_to = config['to_kestrel']['save_to']

        ks_processor = KS_PROCESSOR_REGISTRY[plugin](prototxt,
                                                     caffemodel, b=1,
                                                     n=model_name, v=version,
                                                     p=self.save_to,
                                                     k=parameters_json,
                                                     i=self.input_channel,
                                                     s=self.serialize,
                                                     resize_hw=resize_hw)
        # nnie=nnie)
        ks_processor.process()
        logger.info('save kestrel model to: {}'.format(self.save_to))
        return os.path.join(self.save_to, '{}_{}.tar'.format(self.save_to, version))


@TOKESTREL_REGISTRY.register('det')
class DetToKestrel(object):
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
        model_name = config['to_kestrel'].get('model_name', 'model')
        nnie_cfg = config['to_kestrel'].get('nnie', None)

        if config['to_kestrel'].get('sigmoid', None) is not None:
            self.parameters['sigmoid'] = True
        if not config['to_kestrel'].get('retina_share_location', True):
            self.parameters['retina_share_location'] = False
        if config['to_kestrel'].get('retina_class_first', False):
            self.parameters['retina_class_first'] = True

        version = config['to_kestrel'].get('version', "1.0.0")
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
                                                     n=model_name, v=version,
                                                     p=self.save_to,
                                                     k=parameters_json,
                                                     i=self.input_channel,
                                                     s=self.serialize,
                                                     resize_hw=resize_hw,
                                                     nnie=nnie)
        ks_processor.process()
        # shutil.move(kestrel_model, self.save_to)
        logger.info('save kestrel model to: {}'.format(self.save_to))
        return os.path.join(self.save_to, '{}_{}.tar'.format(self.save_to, version))


@TOKESTREL_REGISTRY.register('cls')
class ClsToKestrel(object):
    def __init__(self, config, caffemodel, parameters, save_to=None, serialize=False, input_channel=3):
        self.config = config
        self.caffemodel = caffemodel
        self.parameters = parameters
        self.save_to = save_to
        self.serialize = serialize
        self.input_channel = input_channel

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
            .format(model_name + '_nnie.tar')
        os.system(tar_cmd)
        logger.info("generate {model_name + '_nnie.tar'} done!")

    def process(self):
        cmd = 'python -m spring.nart.tools.caffe.convert -a {}'.format(self.caffemodel)
        os.system(cmd)
        logger.info('=========merge bn done==========')

        prefix = self.caffemodel.rsplit('.', 1)[0] + '-convert'
        caffemodel = '{}.caffemodel'.format(prefix)
        prototxt = '{}.prototxt'.format(prefix)

        config = copy.deepcopy(self.config)
        nnie_cfg = config['to_kestrel'].get('nnie', None)
        model_name = config['to_kestrel'].get('model_name', 'model')

        version = self.config['to_kestrel'].get('version', '1.0.0')
        to_kestrel_yml = 'temp_to_kestrel.yml'
        with open(to_kestrel_yml, 'w') as f:
            json.dump(self.parameters, f, indent=2)

        parameters_json = 'tmp_parameters_json'
        # convert torch.tensor in parameters to int or list
        for key in self.parameters:
            if torch.is_tensor(self.parameters[key]):
                self.parameters[key] = self.parameters[key].tolist()
        with open(parameters_json, 'w') as f:
            json.dump(self.parameters, f, indent=2)

        if self.save_to is None:
            self.save_to = config['to_kestrel']['save_to']

        logger.info('Converting Model to Kestrel...')
        plugin = config['to_kestrel']['plugin']
        ks_processor = KS_PROCESSOR_REGISTRY[plugin](prototxt,
                                                     caffemodel, b=1,
                                                     n=model_name, v=version,
                                                     p=self.save_to,
                                                     k=parameters_json,
                                                     i=self.input_channel,
                                                     s=self.serialize,
                                                     resize_hw=None,
                                                     nnie=None)
        ks_processor.process()
        logger.info('To Kestrel Done!')

        # convert model to nnie
        nnie_cfg = self.config['to_kestrel'].get('nnie', None)
        if nnie_cfg is not None:
            logger.info('Converting Model to NNIE...')
            self.to_nnie(nnie_cfg, self.config, prototxt, caffemodel, self.save_to)

        logger.info('Save kestrel model to: {}'.format(self.save_to))
        return os.path.join(self.save_to, '{}_{}.tar'.format(self.save_to, version))


@TOKESTREL_REGISTRY.register('det_3d')
class Det3dToKestrel(object):
    def __init__(self, config, caffemodel, parameters, save_to=None, serialize=False, input_channel=3):
        self.config = config
        self.caffemodel = caffemodel
        self.parameters = parameters
        self.save_to = save_to
        self.serialize = serialize
        self.input_channel = input_channel

    def process(self):
        config = copy.deepcopy(self.config)
        plugin = config['to_kestrel']['plugin']

        version = config['to_kestrel'].get('version', "1.0.0")
        parameters_json = 'tmp_parameters_json'
        # convert torch.tensor in parameters to int or list
        for key in self.parameters:
            if torch.is_tensor(self.parameters[key]):
                self.parameters[key] = self.parameters[key].tolist()
        with open(parameters_json, 'w') as f:
            json.dump(self.parameters, f, indent=2)
        resize_hw = config['to_kestrel'].get('resize_hw', '')
        nnie = False

        if self.save_to is None:
            self.save_to = config['to_kestrel']['save_to']

        ks_processor = KS_PROCESSOR_REGISTRY[plugin]('',
                                                     self.caffemodel,
                                                     b=1,
                                                     n=self.save_to,
                                                     v=version,
                                                     p=self.save_to,
                                                     k=parameters_json,
                                                     i=self.input_channel,
                                                     s=self.serialize,
                                                     resize_hw=resize_hw,
                                                     nnie=nnie)
        ks_processor.process()
        logger.info('save kestrel model to: {}'.format(self.save_to))
        return os.path.join(self.save_to, '{}_{}.tar'.format(self.save_to, version))
