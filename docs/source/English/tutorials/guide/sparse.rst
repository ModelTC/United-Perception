Object detection/classification with Msbench
============================================
This part, we introduce how to prune an object detection/classification model using msbench.

Getting Started
-----------------

**1**. **Clone the repositories.**

.. code-block:: python

    
    # Clone UP repository and install it.


**2**. **Sparsification-aware training.**

.. code-block:: python

    # Prepare your float pretrained model.
    cd united-perception/scripts
    # Follow the prompts to set config in train_sparse.sh.
    sh train_sparse.sh


**We have several examples of sat config in united-perception repository:**

For ResNet18-AmbaLevelPruneScheduler:
 - float pretrained config file: configs/cls/resnet/res18.yaml
 - sat config file: configs/cls/resnet/res18_amba_sparse.yaml

For retinanet-AmbaLevelPruneScheduler:
 - float pretrained config file: configs/det/retinanet/retinanet-r18-improve.yaml
 - sat config file: configs/det/retinanet/retinanet-r18-improve_amba_sparse.yaml

For fastercnn-AmbaLevelPruneScheduler:
 - float pretrained config file: /configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml
 - sat config file: /configs/det/faster_rcnn/faster_rcnn_r50_fpn_amba_sparse.yaml

 For ResNet18-AmpereScheduler:
 - float pretrained config file: configs/cls/resnet/res18.yaml
 - sat config file: configs/cls/resnet/res18_ampere_sparse.yaml

For retinanet-AmpereScheduler:
 - float pretrained config file: configs/det/retinanet/retinanet-r18-improve.yaml
 - sat config file: configs/det/retinanet/retinanet-r18-improve_ampere_sparse.yaml

For fastercnn-AmpereScheduler:
 - float pretrained config file: /configs/det/faster_rcnn/faster_rcnn_r50_fpn_1x.yaml
 - sat config file: /configs/det/faster_rcnn/faster_rcnn_r50_fpn_ampere_sparse.yaml


**Something import in config file:**

 - leaf_module: Prevent torch.fx tool entering the module.
 - pretrain_model: The path to your float pretrained model.


**3**. **Resume training during sat.**

.. code-block:: python

    cd united-perception/scripts
    # just set resume_model in config file to your model, we will do all the rest.
    sh train_sparse.sh


**4**. **Evaluate your pruned model.**

.. code-block:: python

    cd united-perception/scripts
    # set resume_model in config file to your model
    # add -e to train_sparse.sh
    sh train_sparse.sh


Introduction of UP-Msbench Project
----------------------------------------

The training codes start in united-perception/commands/train.py.

When you set the runner type to sparse in config file, SparseRunner will be executed in united-perception/up/tasks/sparse/runner/sparse_runner.py.

1. Firstly, build your float model in self.build_model().
2. Load your float pretrained model/pruned model in self.load_ckpt().
3. Use torch.fx to trace your model in self.prepare_sparse_model().
4. Set your optimization and lr scheduler in self.build_trainer().
5. Set some properties for the sparse scheduler in  self.sparse_post_process().
6. Train in self.train()


**Something important:**

 - Your model should be splited into network and post-processing. Fx should only trace the network.
 - We disable the ema in sat. If your ckpt has ema state, we will load ema state into model, as shown in self.load_ckpt().
