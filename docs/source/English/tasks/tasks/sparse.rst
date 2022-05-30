Sparse training
===============

UP supports sparse training;
`Codes <https://github.com/ModelTC/EOD/tree/main/up/tasks/sparse>`_

Configs
-------

It contains the illustration of common configs.

`Repos <https://github.com/ModelTC/EOD/tree/main/configs/sparse>`_

Dataset related modules
-----------------------

1. Dataset types:

  * imagenet
  * custom_cls
  * coco

2. The type of datasets can be chosen by setting 'type' in Dataset. The config is as followed.

  .. code-block:: yaml

    dataset:
      type: cls * or coco
      kwargs:
        ...

3. The following written should refer to the chosen dataset.

Sparse training setting
-----------------------

  .. code-block:: yaml

    runtime:
      runner:
        type: sparse

    sparsity:
      mask_generator:
        type: NormalMaskGenerator
      fake_sparse:
        type: FakeSparse
      scheduler:
        type: AmbaLevelPruneScheduler
        kwargs:
          total_iters: None
          sparsity_table: [30,40,50,60]
          no_prune_keyword: ''
          no_prune_layer: ''
          prun_algo: 1
          prun_algo_tuning: 0.5
          dw_no_prune: False
          do_sparse_analysis: False
          output_dir: path_to/amba/faster_rcnn_r50_fpn_improve_amba_sparse_30_to_90/sparse_analysis
          save_dir: path_to/amba/faster_rcnn_r50_fpn_improve_amba_sparse_30_to_90/sparse_ckpts
      leaf_module: [Space2Depth, FrozenBatchNorm2d]


