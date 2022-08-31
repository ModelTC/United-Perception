发布历史/Release History
========================

v0.3.0
-------

外部依赖版本
^^^^^^^^^^^^

* Spring2 0.6.0
* 集群环境 s0.3.3 / s0.3.4

Breaking Changes
^^^^^^^^^^^^^^^^

* 本次更新取消了部署配置的save_to参数，改为命令行参数设置方式，以tokestrel为例：

  .. code-block:: bash

    python -m up to_kestrel \
      --config=xxx.yaml \
      --save_to=kestrel_model \
      2>&1 | tee log.tokestrel

  默认tokestrel/toadela使用onnx格式，不再默认使用caffe，若希望使用caffe则需手动设置，以检测任务为例：

  .. code-block:: yaml

    to_kestrel:
      toks_type: det
      default_confidence_thresh: 0.05
      plugin: essos   # onnx格式: essos; caffe格式: essos_caffe
      model_name: model
      version: 1.0.0
      resize_hw: 800x1333

* 修改了custom_evaluator中get_miss_rate方法的参数配置，使得能够在其内部计算不同fppi阈值下的ap值。传入时应增加drec与prec，返回值增加fppi_ap

Highlights
^^^^^^^^^^

* 算法类型：
    * 【3D算法】
        * 支持Mono3d 系列算法 FCOS3D、PGD、GUPNet 等算法
    * 【车道线检测】
        * 支持CondlaneNet、GANet 等算法
    * 【目标检测】
        * 支持YOLOV6、OneNet （nms-free）、OTA 等算法
        * 支持重参数化算法RepVgg 版本resnet

* 通用特性
    * 【大模型训练推理】
        * 动态Checkpoint 算法支持：动态的选择最优的checkpoint 模块进行更快速的模型训练 `DC使用说明 <https://confluence.sensetime.com/pages/viewpage.action?pageId=452715805>`
        * MB 系列训练支持：MB4/7/15
        * 超大规模训练list 推理支持：解决了内存溢出的问题、支持断点resume 功能
    * 【神经网络搜索】
        * 支持BigNas 系列模型搜索功能，配合lantency 测速功能可以选取最优的网络结构 `BigNas使用说明 <https://confluence.sensetime.com/pages/viewpage.action?pageId=452714344>`
        * 支持基于强化学习的模型搜索方法 Meta-X
    * 【SenseData集成】
        * 集成SenseData公开数据集，当前coco和imagenet直接使用公共ceph仓库，无需依赖lustre
    * 【模型格式规范化】
        * tokestrel/toadela默认使用onnx格式，不再默认使用caffe。若希望使用caffe则需手动设置

Other Features
^^^^^^^^^^^^^^

* 支持类别数为1时的3d检测及det_3d 任务的部署
* 支持检测标注漏标检查 `#80 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/80>`_
* 公开数据集sensedata ceph 训练支持，目前最常用的coco + imagenet可以直接读取公共ceph仓库
* 支持group evaluator，更好处理密集目标的数据集问题
* 支持video inference功能 `#92 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/92>`_
* 新增inference中分类和分割任务结果可视化的功能 `#75 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/75>`_
* 支持使用onnx模型上传进行部署
* 支持key_replace的weapper方式
* 新增list_balance及class_balance的sampler方式
* 新增异常排查监控工具inspector相关hook
* 支持分割任务中segmentor的kestrel部署
* 新增eval中metric可以指定compare key的功能 `#77 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/77>`_
* 支持忽略指定类别的情况下进行bad case analysis `#78 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/78>`_
* RetinaNet添加share_location, class_first字段，支持每个类的独立回归 （仅支持推理） `#69 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/69>`_
* 新增custom dataset和rank custom dataset的test resume
* 支持在resnet中指定池化层以及out_channel宽度
* 新增分类后处理量化功能
* 支持获取clsdataset中每种类别所包含样本的id及各类别样本数目的方法
* 在caffe_model及onnx_model的doc_string中添加model_hash
* 提供build前后内存使用信息
* 支持计算clseval中平均f1/acc/prec
* 支持MR eval区分目标size进行评测 `#70 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/70>`_
* 支持grad_cam及特征图可视化
* 新增gpu check功能 `#88 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/88>`_
* 部署生成meta.json添加model_name信息
* 新增使用ceph读取sensebee数据示例
* 支持样本标签合理性判断，支持样本漏标可能性的判断
* 支持内存监测相关hook
* 支持根据file list及file folder直接进行推理的功能
* 支持在cfg中指定latest save freq


Bug Fixes
^^^^^^^^^

* 修复了配置文件中data_pool指定为空列表时引起的bug
* 修改分类任务中存储结果，可以选择性存储所有score `#68 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/68>`_
* 修复resnet中freeze layer在参数freeze后mode仍为training的bug `#73 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/73>`_
* 修复typos错误
* 修复了cfg中pretrain_model加载了错误参数的bug
* 修复了saver中拷贝文件及存储ckpt时目标路径存在文件而引起的bug `#94 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/94>`_
* 修复了multitast在eval阶段仍使用sysn bn的bug
* 修复了ckpt中ema值为空时load失败的bug
* 修复了swin_trans cfg文件中lr_scheduler层级错误的bug
* 修复了分类任务中因存储数据变化而引起bad case analysys不适配的bug
* 修复了vis hook 文档和实际参数不匹配的问题 `#76 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/76>`_
* 修复了加载pod-style resnet pretrain时未正确处理ema的问题 `#81 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/81>`_
* 修复了semantic_fpn在inference仍计算loss的bug
* 修复了test_resume中done_imgs变量调用错误的bug
* 修复了world_size为1时与linklink不适配产生的bug
* 修复了retinenet iou分支转模型的bug `#89 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/89>`_
* 修复了QuantRunner类calibrate方法中错误track梯度的问题
* 修复了adela部署时deploy id和benchmark id不匹配的bug
* 修复了inference时读取ckpt时不适配的bug
* 修复了配置文件中train和test使用不同dataloader引起的inference中不适配bug
* 修复了ceph reader读取以‘/’开头的文件时join失败的bug `#83 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/83>`_
* 修复了roi_head中conv前后inplane不适配的bug
* 修复了multicls与kestrel部署不适配的bug
* 修复了检测部署任务中因拆分bbox_head而引起不匹配的bug
* 修复了label_mapping为none时image_source获取错误的bug
* 修复了部署net_graph.leaf大于1时与net_info['score']不匹配的bug
* 修复了launch为mpi时报错的bug


v0.2.0
-------

外部依赖版本
^^^^^^^^^^^^

* Spring2 0.6.0
* 集群环境 s0.3.3 / s0.3.4
* spring_aux-0.6.7.develop.2022_05_07t08_45.333adcd0-py3-none-any.whl

Breaking Changes
^^^^^^^^^^^^^^^^

* 本次重构了检测二阶段的结构组成，为了更加方便的进行量化和稀疏训练, 具体可以参考cfg。
* 修改了模型部署的参数配置。具体cfg 可以从此处查询

  .. code-block:: bash
         
    # 取消了detector参数的使用
    # 常用配置 (以det为例)：
    to_kestrel:
        toks_type: det  # 任务类型
        save_to: KESTREL  # 模型保存路径
        plugin: essos  # kestrel组件
        ...

Highlights
^^^^^^^^^^

* 算法类型：
    * 【3D算法】支持3D Point-Pillar 系列算法, 包含Pointpillar,Second, CenterPoint 等各个算法 `3D benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/benchmark/3d_detection_benchmark.md>`_
    * 【语义分割】支持分割任务最新Sota 算法，Segformer，HrNet 系列，提供超高精度Baseline `Seg benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/benchmark/semantic_benchmark.md>`_
    * 【目标检测】支持最新检测蒸馏算法，大幅度提升模型的精度 benchmark `Det benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/benchmark/distillation.md>`_

* 通用特性：
    * 【Transformer结构】支持Vision Transformer 系列，包含 Swin-Transformer, VIT，CSWin Transformer `Cls benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/benchmark/classification_benchmark.md>`_
    * 【量化与稀疏】支持Amba、Ampere 检测分类稀疏训练 ( `Spring.sparsity <https://confluence.sensetime.com/pages/viewpage.action?pageId=407432119>`_ , `Sparse benchmark <http://spring.sensetime.com/docs/sparsity/benchmark/ObjectDetection/Benchmark.html>`_ )  ；支持TensorRT、Snpe 、VITIS 等多个后端进行QAT量化 ( `spring.quant.online <https://mqbench.readthedocs.io/en/latest/?badge=latest>`_ )，同时支持检测一阶段和二阶段算法 `Quant benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/benchmark/quant_benchmark.md>`_
    * 【自监督算法】支持自监督算法, MOCO 系列、SimClr 系列、simsiam、MAE `SSL benchmark <https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/master/benchmark/ssl_benchmark.md>`_

* 易用工具：
    * 【部署打包自动化】检测、分类、分割、关键点全面支持模型部署打包，支持ADElA 进行模型评测和托管。
    * 【大数据集训练】超大规模数据集训练和测试支持，Rank dataset 扩展到其他任务，同时支持多种模式进行内存友好推理。
    * 【其他】英文文档支持

Other Features
^^^^^^^^^^^^^^

* Condinst FCOS 添加
* 支持通过环境变量进行任务隔离
* 分类任务添加多标签支持和多分类支持
* 支持多个单独测试集eval功能 
* RankDataset 重构支持分类检测等各个任务，支持推理时使用
* 大规模数据集推理内存优化，实时写入磁盘和分组gather 模式
* 提供每个 iteration 耗时统计的分解(数据加载/前处理/forward/backward/梯度allreuce)信息
* 检测支持softer nms
* 新增toonnx 接口，单独支持转换到onnx

Bug Fixes
^^^^^^^^^

* 修复stitch_expand 的没有被注册的bug
* 修复typos 错误
* 修复spconv，numba 引入的显存bug
* 修复各种日志debug 信息输出的bug
* 修复s0.3.3环境不能import InterpolationMode的bug `#23 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/23>`_
* 修复swin和cswin修改尺寸的bug，以进行检测
* 修复condinst中return_pos_inds为True时base_multicls roi_predictor的错误
* 修复推理时加载模型不导入ema的bug
* 修复swin导出不同阶段特征时out_planes不匹配的bug
* 修复cls_dataset meta file 有空格会有问题的bug
* 修复了fp16 grad clipping 的bug
* 修复了推理时有syncbn 报错的bug `#33 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/33>`_
* 修复了单卡测试没有finalize的bug
* 修复了dist 后端出现的一些不适配的bug
* 修复了to kestrel需要初始化linklink与dataset的bug `#22 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/22>`_
* 修复了模型部署不适配不包含post_process的网络结构的bug
* 修复了部署时不能加载ema模型的bug
* 修复了torch_sigmoid_focal_loss设置不同类别alpha的bug
* 支持kitti evaluator自动保存性能最优的模型
* 修复了损失函数不包含模块前缀的bug `#19 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/19>`_
* 修改了adela部署方式，不需要生成release.json
* 修复了gdbp测速不支持多batch size输入的bug
* 支持adela部署设置nart配置参数 `#44 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/44>`_
* 修复了RetinaHead with IoU部署的bug
* 修复了time logger读取环境变量的bug `#57 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/57>`_

Breaking Changes
^^^^^^^^^^^^^^^^

* 本次重构了检测二阶段的结构组成，为了更加方便的进行量化和稀疏训练。具体 `Faster R-CNN <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/det/faster_rcnn>`_ 可以从此处查询。
* 修改了模型部署的参数配置。具体 `Deploy <https://gitlab.bj.sensetime.com/spring2/united-perception/-/tree/master/configs/det/deploy>`_ 可以从此处查询。
    * 取消了detector参数的使用
    * 常用配置 (以det为例)：
        to_kestrel:
          toks_type: det  # 任务类型
          save_to: KESTREL  # 模型保存路径
          plugin: essos  # kestrel组件

v0.1.0
-------

Hightlights
^^^^^^^^^^^^^^^^^^^^^

* 高精度可部署的Baseline，完备的模型生产流程，使用Adela 直接部署模型并进行精度评测。
* 统一的训练任务接口，支持检测，分类，关键点，语义分割等多个任务单独和联合训练。
* 兼容POD 和Prototype 等框架训练的checkpoint 导入，无痛迁移。
* Plugin 开发模式，支持用户自定义模块
* 简便的模型蒸馏方式。
* 统一的训练环境，提供了简便的模型训练接口，用户只需注册少量模块完成新任务训练。
* 统一的文件读取接口，支持ceph + lustre 等各种读取后端。
