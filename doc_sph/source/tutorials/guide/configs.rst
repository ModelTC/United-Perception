配置文件分析 & 注册器模式
=========================

UP sets pipeline parameters by incorporating them into configs.
Considering convenience and expansibility, UP offers common interfaces for all components such as dataset, saver, backbone, etc. which are implemented by REGISTRY Register.

Configuration File
------------------

Standard information includes num_classes, runtime, dataset, trainer, saver, hooks and net. The standard format is as follows:

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


Register Mode
-------------

UP supports register for each module to flexibly compose pipelines.

How to register
---------------

All callable object(inclue function, classes, .etc) can be registered, but the result should be corresponding instance. For example. CocoDataset return a dataset, and  resnet50 return a module(ResNet instance).
The registration format is:
REGISTRY.register(alias).

Registry is corresponding instance, for example, DATASET_REGISTRY, AUGMENTATION_REGISTRY, etc.

The following are examples of registering CocoDataset and resnet50 respectively:

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

How to use Registry
-------------------

UP calls registered modules according to config. Alias should be provided for locating the module.
For example, CocoDataset could be called by type "coco"(alias) and parameters "kwargs".

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


UP Developing Mode
------------------

We highly recommand a novel developing mode called Public UP + Plugins, which composed by two components:

* Public UP: Completed detection frame.
* Plugins: Customized modules registered by REGISTRYs.

Customized Plugin
-----------------

You can develop a plugin which organized by several registered modules, such as datasets, models, losses, etc. Take Face package for instance, the structure is as follows:

  .. code-block:: bash
    
    face
    ├── datasets
    |   └── face_dataset.py
    ├── __init__.py
    └── models
        ├── facenet.py
        └── __init__.py

FaceDataset and FaceNet defined in package need to be registered with DATASET_REGISTRY and MODULE_ZOO_REGISTRY.

Then you need to add package into PLUGINPATH:

  .. code-block:: bash
    
    export PLUGINPATH='path to father_dir_of_face'

UP Development Mode
-------------------

Design described above have following advantages:

    * Import flexibly: After developing a plugin, you only need to add path into PLUGINPATH.
    * Use conveniently: You could origanize pipeline by only adding register aliases into configs. Register details refer to Register.
    * Maintenance friendly: Public UP is well isolated from personal plugins, you only need to maintain your plugins code with few costs.
