Config analysis & Registry
==========================

UP writes the config through setting pipline parameters.
Given the convenience and scalability, UP supports interfaces for all modules including dataset, saver, backbone, and so on.
The interfaces can be used by calling registry.

Configs
-------

A standard config information includes num_classes, runtime, dataset, trainer, saver, hooks and net.
A standard format is as followed.

  .. code-block:: yaml
    
    num_classes: &num_classes xx

    runtime:

    flip:  # transformer: flip, resize, normalize ...
    ......

    dataset:
    ......

    trainer:
    ......

    saver:
    ......

    hooks:
    ......

    net:
    ......


Registry
--------

UP supports registring every module for flexibly constructing experiments.

How to regist
-------------

All named objects, e.g., functions and classes, can be registed, and the registered result should be the corresponding instance.
For example, CocoDataset returns the dataset while resnet50 returns the 'ResNet'.

The registry format: REGISTRY.register(alias).

Registered object should be the corresponding instance.
For example, 'DATASET_REGISTRY'，'AUGMENTATION_REGISTRY', and so on.

Examples of registering CocoDataset and resnet50 are as followed.

  .. code-block:: python
    
    # register a dataset
    @DATASET_REGISTRY.register('coco')
    class CocoDataset(BaseDataset):
        """COCO Dataset"""
        _loader = COCO

        def __init__():
            pass

    # register a backbone
    @MODULE_ZOO_REGISTRY.register('resnet50')
    def resnet50(pretrained=False, **kwargs):
        """
        Constructs a ResNet-50 model.

        Arguments:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        return model

How to use registered instances
-------------------------------

UP uses registered instances through corresponding configs.
Locating registered instances needs the alias of them.
For example, CocoDataset can be used with the alias 'coco' and the parameter 'kwargs'.

  .. code-block:: yaml
    
    dataset: # Required.
    train:
        dataset:
        type: coco  # <------------ alias for COCO
        kwargs:
            meta_file: instances_train2017.json
            image_reader:
            type: fs_opencv
            kwargs:
                image_dir: mscoco2017/train2017
                color_mode: RGB
            transformer: [*flip, *resize, *to_tensor, *normalize]


UP development mode
-------------------

We strongly recommand a new development mode: Public UP + Plugins.

* Public UP: a complete detection framework.
* Plugins: registered costom modules.


User code repository
--------------------

You can develop an user code repository which is built by registered multiple modules such as datasets, models, loss functions, and so on.

  .. code-block:: bash
    
    face
    ├── datasets
    |   └── face_dataset.py
    ├── __init__.py
    └── models
        ├── facenet.py
        └── __init__.py

FaceDataset and FaceNet in package should be registered by 'DATASET_REGISTRY' and 'MODULE_ZOO_REGISTRY', respectively.
Then you should add the package to the path of 'PLUGINPATH' as followed.

  .. code-block:: bash
    
    export PLUGINPATH='path to father_dir_of_face'

The model has the following advantages:
    * Flexible importing: you only need to add the path into 'PLUGINPATH' after developing a plug-in.
    * Conveniently using: you can contrust the calling path through adding alias into configs.
    * Friendly maintaining: Public UP are totally independent with Plugin, and thus you only need to maintain your own codes.
