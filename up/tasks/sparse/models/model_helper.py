# Standard Library
import copy

# Import from third library
import torch
import torch.nn as nn
from up.utils.env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import (
    MODULE_ZOO_REGISTRY,
    MODULE_PROCESS_REGISTRY,
    MODEL_HELPER_REGISTRY
)
from up.models.model_helper import ModelHelper


__all__ = ['SparseModelHelper']


@MODEL_HELPER_REGISTRY.register('sparse')
class SparseModelHelper(ModelHelper):
    """Build model from cfg"""
    def __init__(self, config):
        # Skip the initialization process of ModelHelper and initialize nn.Module directly !
        nn.Module.__init__(self)
        self.model_cfg = copy.deepcopy(config)
        self.config = config
        self.model_list = []
        self.model_map = {}
        self.build_model()
        self.dtype = torch.float32
        self.device = torch.device('cpu')

        self._init_functional()
        self.train()

    def split_cfg(self):
        self.cfg_split_res = []
        cfg_map = {c['name']: c for c in self.config['net']}
        for cfg_subnet in self.config['net']:
            mtype = cfg_subnet['type']
            if env.is_master():
                assert (mtype in MODULE_ZOO_REGISTRY or mtype in MODULE_PROCESS_REGISTRY), \
                    '{} is not supported, avaiables are:{} and {}'.format(mtype,
                                                                          MODULE_ZOO_REGISTRY,
                                                                          MODULE_PROCESS_REGISTRY)
            if cfg_subnet.get('prev', None) is not None:
                if cfg_subnet['prev'] in cfg_map:
                    pre_cfg = cfg_map[cfg_subnet['prev']]
                    if pre_cfg['type'] in MODULE_ZOO_REGISTRY:
                        tmp_module = MODULE_ZOO_REGISTRY.build(pre_cfg)
                    else:
                        tmp_module = MODULE_PROCESS_REGISTRY.build(pre_cfg)
                    if hasattr(tmp_module, "get_outplanes"):
                        cfg_subnet['kwargs']['inplanes'] = tmp_module.get_outplanes()
                        cfg_subnet['prev'] = None
            if len(self.cfg_split_res) == 0:
                if mtype in MODULE_ZOO_REGISTRY:
                    self.cfg_split_res.append(('net', [cfg_subnet]))
                else:
                    self.cfg_split_res.append(('process', [cfg_subnet]))
            elif mtype in MODULE_ZOO_REGISTRY:
                if self.cfg_split_res[-1][0] == 'net':
                    self.cfg_split_res[-1][1].append(cfg_subnet)
                else:
                    self.cfg_split_res.append(('net', [cfg_subnet]))
            else:
                if self.cfg_split_res[-1][0] == 'process':
                    self.cfg_split_res[-1][1].append(cfg_subnet)
                else:
                    self.cfg_split_res.append(('process', [cfg_subnet]))
        # designed for classification task
        if self.cfg_split_res[-1][-1][-1]['type'] == 'base_cls_postprocess':
            post_process_mod_cfg = self.cfg_split_res[-1][-1][-1]
            self.cfg_split_res[-1][-1].remove(post_process_mod_cfg)
            self.cfg_split_res.append(('process', [post_process_mod_cfg]))
        self.split_num = len(self.cfg_split_res)
        logger.debug("self.split_num = {}".format(self.split_num))
        logger.debug("self.cfg_split_res = {}".format(self.cfg_split_res))

    def merge_module_name(self, cfg):
        name_list = [cfg_subnet['name'] for cfg_subnet in cfg]
        merge_name = '_'.join(name_list)
        self.model_map.update({k: merge_name for k in name_list})
        return merge_name

    def build_model(self):
        model_helper_type = self.config['runtime']['model_helper']['type']
        model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
        model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]
        self.split_cfg()
        for cfg in self.cfg_split_res:
            mname = self.merge_module_name(cfg[1])
            module = model_helper_ins(cfg[1], **model_helper_kwargs)
            self.add_module(mname, module)
            if cfg[0] == 'net':
                self.model_list.append(mname)
