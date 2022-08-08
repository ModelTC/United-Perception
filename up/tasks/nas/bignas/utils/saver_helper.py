from up.utils.general.registry_factory import SAVER_REGISTRY, MODEL_HELPER_REGISTRY
from up.utils.general.saver_helper import Saver
from up.utils.general.model_load_utils import load
from up.utils.general.log_helper import default_logger as logger


@SAVER_REGISTRY.register('nas')
class NasSaver(Saver):
    def __init__(self, save_cfg, yml_path=None, work_dir='./'):
        super(NasSaver, self).__init__(save_cfg, yml_path, work_dir)

    def get_subnet_dict(self, subnet):
        subnet_dict = {}
        for item in subnet:
            subnet_dict[item['name']] = item
        return subnet_dict

    def parse_subnet_settings(self, k, subnet_dict):
        output = {}
        if k == 'backbone' or k == 'neck':
            output['kernel_size'] = subnet_dict[k]['kwargs']['kernel_size']
            output['depth'] = subnet_dict[k]['kwargs']['depth']
            output['out_channel'] = subnet_dict[k]['kwargs']['out_channel']
        if k == 'roi_head':
            if 'RPN' in subnet_dict[k]['type']:
                output['cls_subnet'] = {}
                output['cls_subnet']['kernel_size'] = subnet_dict[k]['kwargs']['kernel_size']
                output['cls_subnet']['depth'] = subnet_dict[k]['kwargs']['depth']
                output['cls_subnet']['out_channel'] = subnet_dict[k]['kwargs']['out_channel']
            else:
                output['cls_subnet'] = {}
                output['box_subnet'] = {}
                output['cls_subnet']['kernel_size'] = subnet_dict[k]['kwargs']['kernel_size']
                output['box_subnet']['kernel_size'] = subnet_dict[k]['kwargs']['kernel_size']
                output['cls_subnet']['depth'] = subnet_dict[k]['kwargs']['depth_cls']
                output['cls_subnet']['out_channel'] = subnet_dict[k]['kwargs']['out_channel_cls']
                output['box_subnet']['depth'] = subnet_dict[k]['kwargs']['depth_box']
                output['box_subnet']['out_channel'] = subnet_dict[k]['kwargs']['out_channel_box']
        if k == 'bbox_head':
            output['fc'] = {}
            output['fc']['depth'] = subnet_dict[k]['kwargs']['depth']
            output['fc']['out_channel'] = subnet_dict[k]['kwargs']['out_channel']
        if k == 'cls_head':
            output['out_channel'] = subnet_dict[k]['kwargs']['in_plane']
        return output

    def _adpat_subnet_setting(self, save_cfg):
        subnet_settings = save_cfg['subnet_settings']
        subnet_dict = self.get_subnet_dict(save_cfg['subnet'])
        adapt_settings = {}
        for k, v in subnet_settings.items():
            if v is not None:
                adapt_settings[k] = v
            else:
                adapt_settings[k] = self.parse_subnet_settings(k, subnet_dict)
        return adapt_settings

    def get_subnet_weight(self, super_model, save_cfg=None):
        subnet_settings = self._adpat_subnet_setting(save_cfg)
        subnet = {}
        inplanes = None
        for name, m in super_model.named_children():
            if hasattr(m, 'sample_active_subnet_weights') and name in subnet_settings:
                m.inplanes = inplanes
                subnet[name] = m.sample_active_subnet_weights(subnet_settings[name], inplanes)
                if hasattr(subnet[name], "get_outplanes"):
                    inplanes = subnet[name].get_outplanes()
            else:
                if save_cfg.get('keep_supernet_weight', True):
                    subnet[name] = m
        return subnet

    def load_supernet(self, super_state, strict=False):
        model_helper_ins = MODEL_HELPER_REGISTRY['base']
        super_model = model_helper_ins(self.save_cfg['supernet'])
        logger.info('load supernet state')
        super_model.load(super_state, strict=strict)
        subnet = self.get_subnet_weight(super_model.cuda(), self.save_cfg)
        subnet_model = model_helper_ins(self.save_cfg['subnet'])
        for mname, module in subnet_model.named_children():
            if mname not in subnet:
                continue
            else:
                load(module, subnet[mname].state_dict(), strict=strict)
        return {"model": subnet_model.state_dict()}

    def load_pretrain_or_resume(self):
        if self.auto_resume:
            last_checkpoint_path = self.find_last_checkpoint()
            if last_checkpoint_path is not None:
                logger.warning('Load checkpoint from {}'.format(last_checkpoint_path))
                return self.load_checkpoint(last_checkpoint_path)
            else:
                logger.warning('Not found any valid checkpoint yet')

        if 'resume_model' in self.save_cfg:
            logger.warning('Load checkpoint from {}'.format(self.save_cfg['resume_model']))
            state = self.load_checkpoint(self.save_cfg['resume_model'])
            if 'supernet' in self.save_cfg:
                state = self.load_supernet(state['model'])
            return state
        elif 'pretrain_model' in self.save_cfg:
            state = self.load_checkpoint(self.save_cfg['pretrain_model'])
            logger.warning('Load checkpoint from {}'.format(self.save_cfg['pretrain_model']))
            if 'supernet' in self.save_cfg:
                state = self.load_supernet(state['model'])
                return state
            return {'model': state['model']}
        else:
            logger.warning('Load nothing! No weights provided {}')
            return {'model': {}}
