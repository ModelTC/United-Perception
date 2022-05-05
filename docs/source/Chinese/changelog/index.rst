发布历史/Release History
========================

v0.2.0
-------

外部依赖版本
^^^^^^^^^^^

* Spring2 0.6.0
* 集群环境 s0.3.3 / s0.3.4
* spring_aux-0.6.7.develop.2022_05_07t08_45.333adcd0-py3-none-any.whl

Breaking Changes
^^^^^^^^^^^^^^^^

* 本次重构了检测二阶段的结构组成，为了更加方便的进行量化和稀疏训练。具体cfg 可以从此处查询
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

* 支持3D Point-Pillar 系列算法, 包含Pointpillar,Second, CenterPoint 等各个算法
* 支持Vision Transformer 系列，包含 Swin-Transformer, VIT，CSWin Transformer
* 支持分割任务最新Sota 算法，Segformer，HrNet 系列，提供超高精度Baseline
* 支持最新检测蒸馏算法，大幅度提升模型的精度 benchmark
* 支持稀疏训练
* 支持量化 QAT, PTQ
* 支持自监督算法, MOCO 系列、SimClr 系列、simsiam、MAE
* 支持多任务训练
* 检测、分类、分割、关键点全面支持模型部署打包，支持ADElA 进行模型评测和托管。
* 超大规模数据集训练和测试支持，Rank dataset 扩展到其他任务，同时支持多种模式进行内存友好推理。
* 中英文文档支持

New Features
^^^^^^^^^^^^

* Condinst FCOS 添加
* 添加ceph数据使用示例 `#16 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/16>`_
* 支持通过环境变量进行任务隔离
* 添加voxel_generator替代spconv中的接口
* 支持kitti数据集的ceph读取方式
* 分类任务添加多标签支持和多分类支持
* 支持多个单独测试集eval功能 `#8 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/8>`_
* RankDataset 重构支持分类检测等各个任务，支持推理时使用
* 大规模数据集推理内存优化，实时写入磁盘和分组gather 模式
* 提供每个 iteration 耗时统计的分解(数据加载/前处理/forward/backward/梯度allreuce)信息
* 支持softer nms
* Toonnx 接口单独支持
* 添加CI测试并提供各任务部署示例 `#29 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/29>`_
* 多分类任务添加mixup与cutmix数据增广支持
* 支持检测与分类任务的错例分析 `#36 <https://gitlab.bj.sensetime.com/spring2/united-perception/-/issues/36>`_

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
