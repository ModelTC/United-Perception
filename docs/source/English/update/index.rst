UPdate
======

For showing the latest updates of UP.


Algorithms
----------

* Add quant models of Faster R-CNN, such as 'configs/det/deploy/faster_rcnn_r50_C4_1x_harpy.yaml'.

* Add 'hrnet' series, such as 'configs/seg/hrnet/hrnet18_1024x1024.yaml'.

* Add 'second' series, such as 'configs/det_3d/centerpoint/centerpoint_second.yaml'.

* Add 'segformer' series, such as 'configs/seg/segformer_b0.yaml'.


Datasets
--------

* Add 'rank_cls' dataset in 'cls_dataset'.

* Revise the style of 'mask reader' in 'seg_dataset'.

* Add the 3D dataset as 'kitti'.


Benchmark
---------

* Add the 'sparse_benchmark'.

* Update some data in '3d_detection_benchmark'.

* Add 'segformer' and 'hrnet' benchmarks.

* Add the 3D detection benchmark.


Bugs
----

* Refactor 'rank_dataset' to support test.

* Refactor 'rfcn'.

* Fix bugs in 'cls_evaluator'.

* Fix the bugs of torch syncbn in 'bn_helper'.

* Fix the bugs of mimic configs and 'sample_feature'.

* Fix the bugs of loading model with 'ema'.

* Fix the bugs of FRS in 'mimicker'.

* Fix the bugs of 'multitask' for supporting 'ddp' mode.

* Fix the bugs of 'numba' for 3D detection.

* Fix the bugs in 'data_loader' and 'seg_dataloader' for checking the shape of labels.

* Fix the bugs in 'adela_helper' for adding deploy.

* Fix the bugs for offline running of segmentation by adding 'try'. 

* Fix the bugs in 'roi_supervisor' for removing empty cache.

* Fix the bugs in 'custom_evaluator'.

* Fix the bugs in 'seg_dataloader'.

* Fix the bugs of 'vis_gt' and 'vis_dt' hooks.

* Fix the multiple batch size in cls task in environment of 's0.3.4'.

* Fix the spconv import bugs in 3D detection.

* Fix the bugs in ddp timer.

* Fix the bugs in 'color_jitter' as followed.

  .. code-block:: bash

     color_jitter: &color_jitter
       type: color_jitter_mmseg
       kwargs:
         color_type: &color_type RGB

     dataset:
       train:
         ...
         color_mode: *color_type

* Move runners to tasks.

* Fix the bug of the 'base_multicls' in 'roi_predictor' when 'return_pos_inds' is True.

* Use finalize to fix the bugs coming with 'setup_distributed'.

* Revise the backbone 'cswin' for detection.

* Fix bugs of 'mimic' kwargs.


Features
--------

* Add configs for self supervise learning.

* Support self supervise learning. 

* Add 'group all_gather' for inference.

* Regroup configs of 'seg'.

* Add pretrained models of 'hrnet'.

* Sport 'ema' for 'multitask'.

* Add the script of quant as 'train_qat.sh'.

* Add configs of quant.

* Add quant ptq support.

* add memory friendly inference in 'base_runner'.

* Add export for IoUPostProcess.

* Add softer NMS.

* Support sparse training.

* Support reading 'config_json' which is a json file.

* Add read time and transform time in dataset.

* Add multi-platform deployment && benchmark && publish with 'to_adela', and is shown in 'configs/det/deploy/retinanet-r50_1x_essos_adela.yaml' as followed.
  
  .. code-block:: bash

     - platform: &platform1 'cuda10.0-trt7.0-int8-T4'
       max_batch_size: &max_batch_size1 2
       quantify: True
       quantify_dataset_name: 'faces_simple_quant'
     - platform: &platform2 'cuda10.0-trt7.0-int8-T4'
       max_batch_size: &max_batch_size2 2
       quantify: True
       quantify_dataset_name: 'faces_simple_quant'
     - platform: &platform3 'cuda10.0-trt7.0-int8-T4'
       max_batch_size: &max_batch_size3 2
       quantify: True
       quantify_dataset_name: 'faces_simple_quant'

* Add sub-command to convert model with torch onnx as followed.

  .. code-block:: bash
    
    spring.submit run -n$1 -p spring_scheduler --gpu --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
    "python -m up to_onnx \
      --config=$cfg \
      --save_prefix=toonnx \
      --input_size=3x512x512 \
      2>&1 | tee log.toonnx.$T.$(basename $cfg) "

* Add the function of respectively evaluating datasets.

* Supports 'pytorch.convert.convert_v2'.

* Add the 'batch_time' breakdown timer.
