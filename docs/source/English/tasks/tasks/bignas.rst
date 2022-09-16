.. _NasAnchorEng:

BigNas
=========

NAS is a common method to adjust the network structure in deep learning. It can greatly improve the performance of existing manually designed networks. Using the bignas module in up can help you quickly get this accuracy improvement.
Bignas in up supports the whole process of model training, search and finetune subnet of classification and detection tasks.

Bignas characteristics in up
-----------------------------

* Support a variety of searches that affect the performance of the network structure, such as depth, out_channel, kernel_size, group number, expand_ratio
* A variety of dynamic block implementations, including DynamicBasicBlock, DynamicConvBlock, DynamicRegBottleneckBlock, DynamicLinearBlock, etc 
* Only need to add a small amount of code in the plugin mode, you can customize the supnet and use the following training, search and other functions 
* Support model search for classification and detection. Corresponding examples are provided in up, which can be run with one click

Configuration File
-------------------

    * `Train_supnet <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_train_supnet.yaml>`_
    * `Evaluate_subnet <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_evaluate_subnet.yaml>`_
    * `Finetune_subnet <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_finetune_subnet.yaml>`_
    * `Sample_flops <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_sample_flops.yaml>`_
    * `Sample_accuracy <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_sample_accuracy.yaml>`_
    * `Load the weight of pretrained supnet <https://github.com/ModelTC/United-Perception/blob/main/configs/nas/bignas/det/bignas_retinanet_R18_subnet.yaml>`_

Get Started
--------------

1. Modify the bignas field in the cfg:

    In the bignas field, you can adjust relevant settings such as supnet training, sampling subnet, distill, speed measurement, etc

    .. code-block:: yaml

        bignas:
            train: # This field is used to write information related to supnet training
                sample_subnet_num: 1
                sample_strategy: ['max', 'random', 'random', 'min']
                valid_before_train: False # Is there a verification before the supnet training
            data:  # This field writes information related to the input resolution adjustment
                share_interpolation: False
                interpolation_type: bicubic
                image_size_list: [[1, 3, 768, 1344]]
                calib_meta_file: /mnt/lustre/share/shenmingzhu/calib_coco/coco_2048_coco_format.json
                metric1: bbox.AP
                metric2: bbox.AP.5
            distiller:  # This field is used to write information related to distill. It supports multi_teacher and multi_task distill mode
                mimic:
                    mimic_as_jobs: True
                    mimic_job0:
                    loss_weight: 0.5
                    teacher0:
                        mimic_name: ['neck.p6_conv.conv']
                    student:
                        mimic_name: ['neck.p6_conv.conv']
                teacher0: {} # If the teacher model is supnet, give {} here. You can also customize the teacher model. Please refer to the up/distill section for specific usage
            subnet: # Do not add this field when training supnet. This field is only required when evaluating, finetune, sample flops & acc
                calib_bn: True 
                image_size: [1, 3, 768, 1344]
                flops_range: ['200G', '400G']  # Specify the range of flops when sampling subnets, and you can conduct a round of coarse screening through flops
                baseline_flops: 200G # it is needed when calculating Pareto
                subnet_count: 100  # Number of sampling nets
                subnet_sample_mode: traverse # Set traverse mode to traverse the model structure within the search interval by stride
                subnet_settings: # This field needs to be specified when testing a single subnet
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
                save_subnet_prototxt: False # Whether to crop the weight of the subnet from the supnet

2. Adjust the network and its search_space

    .. code-block:: yaml

        net:
        - name: backbone            
            type: big_resnet_basic
            kwargs:
            ···
            normalize:
                type: dynamic_solo_bn # Dynamic normalize, supports dynamic_sync_bn and dynamic_solo_bn
            out_channel: # Define the searchspace of the backbone part，and specify the search upper limit, search lower limit, sampling strategy and other parameters of out_channel and depth
                space:
                    min: [32, 48, 96, 192, 384]
                    max: [64, 80, 160, 320, 640]
                    stride: [16, 16, 32, 64, 128]
                sample_strategy: stage_wise # sampling strategy between maximum and minimum values，supports stage_wise、stage_wise_depth、block_wise, etc
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

    Bignas has implemented the construction of a variety of dynamic modules, which can be used to customize the network and specify the search space. Users can also customize dynamic modules to build networks according to their needs (it is recommended to use the plugin mode to import, which is always easy to use). The custom network structure needs to inherit the BignasSearchSpace class. For details, please refer to the construction of BigResNetBasic and other network structures

3. Training Supnet

    In the process of Supernet training, the model will be adjusted through the function adjust_model. adjust_model is necessary in the process of model training

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

4. Sample Supnet

    * Evaluate_Subnet
        We will have a target network called base model before searching. After the supnet training, we can first see the accuracy of this base model. We only need to write the right subnets, and then we can measure the speed and test the network
    * Finetune_Subnet
        If the accuracy of the selected model is not up to the standard, or we hope it is higher, we can continue to use a small amount of epoch to finetune, and the LR generally needs to be reduced to about one tenth of the original supnet training
    * Sample_FLOPs
        Random sampling subnet in the net, you can specify the range of flops of the sample subnet. Measure and print the flops, para and speed of the sampled subnet. In the range of all flops in the supnet, about 1W subnets are randomly sampled for visualization, and the distribution map of flops in the current supnet can be obtained
    * Sample_Accuracy
        Random sampling subnet in the net, you can specify the range of flops of the sample subnet. About 2K models can be selected.

5. Test subnet

    * Before training supnet, we can test the flops of the supnet. For example, we can test the distribution of flops in the supernet and whether there are many subnets near the baseline_flops
