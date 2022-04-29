import json
from up.utils.general.petrel_helper import PetrelHelper
from up.tasks.cls.data.cls_dataset import BaseParser, CLS_PARSER_REGISTRY


@CLS_PARSER_REGISTRY.register('moco_imagenet')
class MoCoParser(BaseParser):
    def parse(self, meta_file, idx, metas, start_index=0, rank_indices=None):
        with PetrelHelper.open(meta_file) as f:
            for index, line in enumerate(f):
                if rank_indices is not None:
                    if (start_index + index) not in rank_indices:
                        continue
                cls_res = {}
                filename, label = line.strip().split()
                cls_res['filename'] = filename
                cls_res['label'] = 0
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas


@CLS_PARSER_REGISTRY.register('moco_custom')
class CustomMoCoParser(BaseParser):
    def parse(self, meta_file, idx, metas, start_index=0, rank_indices=None):
        with PetrelHelper.open(meta_file) as f:
            for index, line in enumerate(f):
                if rank_indices is not None:
                    if (start_index + index) not in rank_indices:
                        continue
                cls_res = {}
                res = json.loads(line.strip())
                filename = res['filename']
                cls_res['label'] = 0
                cls_res['filename'] = filename
                cls_res['image_source'] = idx
                metas.append(cls_res)
        return metas
