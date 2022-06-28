import torch
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import DEPLOY_REGISTRY
from up.tasks.quant.runner import QuantRunner
from mqbench.utils.state import enable_quantization
from up.utils.general.global_flag import DEPLOY_FLAG


__all__ = ['QuantDeploy']


@DEPLOY_REGISTRY.register('quant')
class QuantDeploy(QuantRunner):
    def __init__(self, config, work_dir='./'):
        self.config = config
        super(QuantDeploy, self).__init__(config, work_dir, False)

    def build(self):
        self.build_ema()
        self.build_saver()
        self.load_ckpt()
        self.build_model()
        self.get_onnx_dummy_input()
        self.quantize_model()
        self.resume_model_from_quant()
        enable_quantization(self.model)

    def resume_model_from_quant(self):
        self.model.load_state_dict(self.ckpt['model'])

    def get_onnx_dummy_input(self):
        self.model.cuda().eval()
        DEPLOY_FLAG.flag = False
        self.build_dataloaders()
        self.build_hooks()
        batch = self.get_batch('test')
        output = self.model(batch)
        self.dummy_input = {k: v for k, v in output.items() if torch.is_tensor(v)}
        DEPLOY_FLAG.flag = True

    def deploy(self):
        logger.info("deploy model")
        from mqbench.convert_deploy import convert_deploy
        deploy_backend = self.config['quant']['deploy_backend']
        self.model.cuda().eval()
        print('ONNX input shape is: ', self.dummy_input['image'].shape)

        for index, mname in enumerate(self.model_list):
            mod = getattr(self.model, mname)
            print('{}/{} model will be exported.'.format(index + 1, len(self.model_list)))
            print('Model name is : ', mname)
            print()
            convert_deploy(model=mod,
                           backend_type=self.backend_type[deploy_backend],
                           dummy_input=self.dummy_input,
                           model_name=mname)
