.. _NasAnchor:

模型搜索
=========

神经网络结构搜索是深度学习中常用的一种调整网络结构的手段，可以对已有的手工设计的网络带来极大的性能提升，使用up中bignas模块可以帮助大家快速得到这种精度提升。
Up中BigNAS支持分类及检测任务模型训练、搜索、finetune子网的全部流程。

up中bignas特性
--------------

* 支持多种影响网络结构性能的搜索，比如depth，out_channel，kernel_size，group number，expand_ratio
* 多种动态的block实现，包括DynamicBasicBlock，DynamicConvBlock，DynamicRegBottleneckBlock，DynamicLinearBlock 等 
* 只需在以plugin模式加入少量的代码，即可以自定义超网并可使用后续的训练，搜索等功能
* 支持分类和检测的模型搜索，在up里面提供了对应的范例，可以一键运行

配置文件
--------

    * `超网训练 <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_train_supnet.yaml>`_
    * `子网测试 <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_evaluate_subnet.yaml>`_
    * `子网finetune <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_finetune_subnet.yaml>`_
    * `子网采样测Flops <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_sample_flops.yaml>`_
    * `子网采样测试精度 <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_sample_accuracy.yaml>`_
    * `load超网pretrain权重 <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_subnet.yaml>`_

Get Started
--------------

1. 修改配置文件中bignas字段:

    在bignas字段中可对超网训练，采样子网，蒸馏，速度测量等相关设置进行调整

    .. code-block:: yaml

        bignas:
            train: # 这个字段写跟超网训练相关的信息
                sample_subnet_num: 1
                sample_strategy: ['max', 'random', 'random', 'min']
                valid_before_train: False #在超网训练前是否进行一次验证
            data:  # 这个字段写跟input resolution调整相关的信息
                share_interpolation: False
                interpolation_type: bicubic
                image_size_list: [[1, 3, 768, 1344]]
                calib_meta_file: /mnt/lustre/share/shenmingzhu/calib_coco/coco_2048_coco_format.json
                metric1: bbox.AP
                metric2: bbox.AP.5
            distiller:  # 该字段写蒸馏相关信息，如不需要可不添加该字段，支持多教师多任务蒸馏模式
                mimic:
                    mimic_as_jobs: True
                    mimic_job0:
                    loss_weight: 0.5
                    teacher0:
                        mimic_name: ['neck.p6_conv.conv']
                    student:
                        mimic_name: ['neck.p6_conv.conv']
                teacher0: {} # 若教师网络为超网，此处给{}。也可自定义教师网络，具体使用方法可参照up/distill部分
            subnet: # 训练超网时不要加该字段，该字段只在evaluate，finetune，sample flops&acc的时候需要
                calib_bn: True 
                image_size: [1, 3, 768, 1344]
                flops_range: ['200G', '400G']  # 采样多个子网的时候指定flops的range，可以通过flops进行一轮粗筛
                baseline_flops: 200G # 在计算pareto的时候需要
                subnet_count: 100  # 需要采样子网的数目
                subnet_sample_mode: traverse # 设置traverse模式可按stride遍历搜索区间内的模型结构
                subnet_settings: # 在测试单个子网的时候需要指定这个字段
                    backbone:
                        kernel_size: [3, 3, 3, 3, 3]
                        out_channel: [32, 48, 96, 192, 384]
                        depth: [1, 1, 2, 2, 4]
                    neck:
                        depth: [3, 5]
                        out_channel: [128, 128]
                        kernel_size: [1, 3]
                    roi_head.cls_subnet:
                        depth: [4]
                        out_channel: [64]
                        kernel_size: [3]
                    roi_head.box_subnet:
                        depth: [4]
                        out_channel: [64]
                        kernel_size: [3]
                save_subnet_prototxt: False # 从超网中crop出子网的权重

2. 调整网络及其搜索空间

    .. code-block:: yaml

        net:
        - name: backbone            
            type: big_resnet_basic
            kwargs:
            ···
            normalize:
                type: dynamic_solo_bn # 动态模块normalize 支持dynamic_sync_bn以及dynamic_solo_bn
            out_channel: # 定义backbone部分的searchspace，指定out_channel以及depth的搜索上限、搜索下限及采样策略等参数
                space:
                    min: [32, 48, 96, 192, 384]
                    max: [64, 80, 160, 320, 640]
                    stride: [16, 16, 32, 64, 128]
                sample_strategy: stage_wise # sample_strategy是在最大值和最小值之家的采样策略，支持stage_wise、stage_wise_depth、block_wise等
            kernel_size:
                space:
                    min: [3, 3, 3, 3, 3]
                    max: [7, 3, 3, 3, 3]
                    stride: 2
                sample_strategy: stage_wise
            expand_ratio: [0.5, 1, 1, 1, 1]
            depth:
                space:
                    min: [1, 1, 2, 2, 4]
                    max: [1, 3, 4, 4, 6]
                    stride: [1, 1, 1, 1, 1]
                sample_strategy: stage_wise_depth

    bignas中已实现多种动态模块的构建，可通过这些动态模块自定义网络并指定搜索空间。用户也可以根据需求自定义动态模块来构建网络（此处建议使用plugin模式进行导入，灰常好用）。
    自定义网络结构需要继承BignasSearchSpace类，具体可参考BigResNetBasic等网络结构的构建

3. 超网训练

    在超网训练过程中，会通过adjust_model()函数对模型进行调整，adjust_model在模型训练过程中是必要的

    .. code-block:: python

        for iter_idx in range(self.start_iter, self.max_iter):
            batch = self.get_batch('train')
            self.model.zero_grad()
            for curr_subnet_num in range(self.controller.sample_subnet_num):
                self.curr_step = iter_idx
                self.curr_subnet_num = curr_subnet_num
                self.controller.adjust_teacher(batch, curr_subnet_num)
                self.adjust_model(batch)
                task_loss, output = self.forward_train(batch)
                output.update({'cur_iter': iter_idx})
                mimic_loss = self.controller.get_distiller_loss(self.sample_mode, output, curr_subnet_num)
                self._hooks('after_forward', self.cur_iter, output)
                loss = mimic_loss + task_loss
                self.backward(loss)
            self.update()
            self.lr_scheduler.step()

4. 超网sample

    * Evaluate_Subnet
        一般来说，我们在搜索之前会有一个对标的网络，叫做base model，比如我们是基于resnet18来进行搜索的，那么在超网训练好之后，我们可以首先看看这个base model的精度是多少，我们只需要把相当于subnet_settings写好，就可以对这个网络进行测速，测试精度
    * Finetune_Subnet
        如果我们选出来的模型的精度还没有达标，或者还希望更高，可以直接对这个模型进行finetune少量的epoch，lr一般需要减少到原来超网训练的十分之一左右
    * Sample_FLOPs
        随机采样网络中的子网，可以指定sample子网的FLOPs的range, 对采样得到子网的Flops,para以及速度进行测量并打印，在超网的所有的FLOPs范围内随机采样1w左右子网可视化可以得到当前超网的FLOPs的分布图，确保我们搜索的网络是在分布图的尖峰附近
    * Sample_Accuracy
        随机采样网络中的子网，可以指定sample子网的FLOPs的range，一般来说可以选2k个左右。

5. 子网测试

    * 在超网训练之前，我们可以对超网的FLOPs进行测试，比如说，我们可以测试这个超网中FLOPs的分布情况，是否在baseline_flops附近有非常多的子网。
