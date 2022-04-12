import json
from easydict import EasyDict
from up.utils.general.petrel_helper import PetrelHelper
from up.tasks.cls.data.cls_dataset import BaseParser, CLS_PARSER_REGISTRY


@CLS_PARSER_REGISTRY.register('moco_imagenet')
class MoCoParser(BaseParser):
    def parse(self, meta_file, idx, metas):
        with PetrelHelper.open(meta_file) as f:
            for line in f:
                cls_res = {}
                filename, label = line.strip().split()
                cls_res['filename'] = filename
                cls_res['label'] = 0
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas


@CLS_PARSER_REGISTRY.register('moco_custom')
class CustomMoCoParser(BaseParser):
    def parse(self, meta_file, idx, metas):
        with PetrelHelper.open(meta_file) as f:
            for line in f:
                cls_res = {}
                res = json.loads(line.strip())
                filename = res['filename']
                if isinstance(res['label'], list):   # support multilabel cls
                    cls_res['label'] = [int(i) for i in res['label']]
                else:
                    cls_res['label'] = 0
                cls_res['filename'] = filename
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas
