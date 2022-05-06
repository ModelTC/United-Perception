稀疏训练
========

UP支持稀疏训练;
`具体代码 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/up/tasks/sparse>`_

配置文件
--------

`代码仓库 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/sparse>`_
其中包括常用算法配置文件

数据集相关模块
--------------

1. 数据集类型包括:

  * imagenet
  * custom_cls
  * coco

2. 数据集类型通过设置Dataset的type来选择，配置文件示例如下:

  .. code-block:: yaml

    dataset:
      type: cls * 或者 coco
      kwargs:
        ...

3. 后续数据集写法同选择类型。

稀疏训练设置
------------

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


