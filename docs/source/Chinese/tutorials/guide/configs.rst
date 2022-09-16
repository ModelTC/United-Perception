配置文件分析 & 注册器模式
=========================

UP通过设置流程参数来设置配置文件。
考虑到便利性和可延展性，UP为所有组建提供通常接口，包括：数据库(dataset)，存储器(saver)，和骨干网络(backbone)等。
可以通过调用注册器来使用这些接口。

配置文件
--------

标准的配置信息包括：
num_classes, runtime, dataset, trainer, saver, hooks and net. 
标准的形式如下所示:

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


注册器模式
----------

UP 支持对每个模块注册来灵活的构建实验。


如何注册
--------

所有的有名称的对象（函数和类等）都是可以被注册的，但结果应当是对应的实例。
比如，CocoDataset 将返回数据集而 resnet50 返回模块（ResNet instance）。

注册的形式是：
REGISTRY.register(alias).

注册体是相应的实例，比如：DATASET_REGISTRY，AUGMENTATION_REGISTRY 等。

下面分别有注册 CocoDataset 和 resnet50 的例子。

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

怎么使用注册体
--------------

UP 通过对应的配置文件使用注册体。定位注册体则需要使用别名。
例如，CocoDataset 可以通过别名 coco 和参数 "kwargs" 来使用。

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


UP 开发模式
-----------

我们强烈推荐一种新的开发模式： Public UP + 插件（Plugins），包含以下两部分：

* Public UP： 完整的检测框架。
* 插件： 经过注册的自定义模块。


用户代码仓库
------------

你可以开发一个由注册过的多模块组成的用户代码仓库，比如数据集、模型、损失函数等。
以 Face package 为例，结构如下所示。

  .. code-block:: bash
    
    face
    ├── datasets
    |   └── face_dataset.py
    ├── __init__.py
    └── models
        ├── facenet.py
        └── __init__.py

package 中的 FaceDataset 和 FaceNet 应当分别由 DATASET_REGISTRY 和 MODULE_ZOO_REGISTRY 注册。

然后你需要将 package 加入 PLUGINPATH 的路径：

  .. code-block:: bash
    
    export PLUGINPATH='path to father_dir_of_face'

这种模式有以下的优势：
    * 灵活导入： 在开发了一个插件后，您仅需要将路径加入 PLUGINPATH。
    * 使用方便： 您可以仅通过将别名加入配置文件的方式来构建调用路径。
    * 维护友好： Public UP 是和Plugin完全独立的，您可以仅需要花费少量精力来维护您的代码。

