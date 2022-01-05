from .python.roi_align import RoIAlignPool


def build_generic_roipool(pool_cfg):
    """
    Construct generic RoIPooling-like class

    Arguments:
        pool_cfg: `method` key to speicify roi_pooling method,
            other keys is configuration for this roipooling `method`
    """
    ptype = pool_cfg['method']
    pool_cls = {
        'roialignpool': RoIAlignPool
    }[ptype]
    # RoIAlignPool
    return pool_cls.from_params(pool_cfg)
