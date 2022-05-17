Data normalization
==================

Normalization supports 5 modes.

* freeze_bn: fixed mean and var.

  .. code-block:: yaml

    normalize:
      type: freeze_bn

* solo_bn: seperately counting mean and var by the single GPU.

  .. code-block:: yaml

    normalize:
      type: solo_bn

* pt_sync_bn: synchronizing mean and var by multiple GPUs by pytorch.

  .. code-block:: yaml

    normalize:
      type: pt_sync_bn
      kwargs:
        group_size: 8

* gn: Group Normalization

  .. code-block:: yaml

    normalize:
      type: gn
      kwargs:
        num_groups: 32

* caffe_freeze_bn: using the frozen bn preloaded from caffe.

  .. code-block:: yaml

    normalize:
      type: caffe_freeze_bn
