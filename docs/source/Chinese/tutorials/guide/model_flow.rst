模型配置与自定义
================

* :ref:`BackboneNeckAnchor`
* :ref:`HeadAnchor`

模型由若干子模块构成，子模块内部实现互相独立，子模块之间的依赖由约定的接口实现；
模块之间的输入输出均为字典类型

任何模型结构可以被抽象为“特征提取模块 + 任务分支”的结构，
特征提取模块包括 backbone 和 neck，任务分支则是针对不同子任务具有不同的结构

.. _BackboneNeckAnchor:

**BackBone & Neck**
~~~~~~~~~~~~~~~~~~~

1. 任何的Backbone或者Neck需要继承于 :class:`torch.nn.Module` , 需要实现以下几个接口:

  * :meth:`~up.models.backbones.ResNet.__init__` 当模块有前驱时，第一个参数为前趋模块输出channel数
  * :meth:`~up.models.backbones.ResNet.get_outplanes` 当模块有后继时，需实现此方法返回该模块的输出channel数，帮助构建后继网络
  * :meth:`~up.models.backbones.ResNet.forward` input为dataset的输出，该方法的输出格式一般为

  .. code-block:: python

    {'features':[], 'strides': []}

2. :meth:`__init__` 函数的参数主要取决于配置文件, 例如，我们先使用一个最简单的配置文件 `custom_net.yaml`, 实现一个L层的卷积网络.

  .. note::

    确保config文件中定义的type为能够被import的模块，在这个例子中，我们新建一个文件custom.py放在 `up.models.backbones` 文件夹下面.
    可以使用注册器来注册新模块并指定一个别名，在配置文件中指定别名即可使用该模块。

  .. code-block:: yaml

    net:
        name: backbone
        type: custom_net
        kwarg:
            depth: 3
            out_planes: [64, 128, 256]


  .. code-block:: python

    import torch
    import torch.nn as nn
    
    class CustomNet(torch.nn.Module):
        def __init__(self, depth, out_planes):
            """
            构造参数为配置文件中的kwarg, 在这个例子中由于没有前驱模块，
            所以没有inplances参数
            """
            self.out_planes = out_planes
          
            in_planes = 3
            for i in range(depth):
                self.add_module(f'layer{i}',
                                nn.Conv2d(in_planes, out_planes[i], kernel_size=3, padding=1))
                self.add_module('relu', nn.ReLU(inplace=True))
                in_planes = out_planes[i]

然后我们再实现 :meth:`forward` 和 :meth:`get_outplanes` 函数

  .. note::

    :meth:`foward` 函数需要计算输出的features和strides, 这两个值都为数组形式。

  .. code-block:: python

    def forward(self, input):
        """
        input的字典类型，数据的组织方式主要取决于config中定义好的Dataset, 
        在这里我们假设input中包含了image这一项
        """

        x = input['image']

        for submodule in self.children():
            x = submodule(x)

        # 输出为一个字典，需要包括features和strides两项, 同时我们保留input中的其他数据
        input['features'] = [x]
        input['strides'] = [1]

        return input

    def get_outplanes(self):

        return self.out_planes

  .. note::
    
    对于backbone，可以在__init__.py中引用该类，会自动注册至MODULE_ZOO_REGISTRY；
    对于检测与分割任务中可能用到的neck，需要通过@MODULE_ZOO_REGISTRY.register("bias")将对应的类注册至MODULE_ZOO_REGISTRY；

.. _HeadAnchor:

**Head**
~~~~~~~~

1. Head模块需要继承-:class:`torch.nn.Module` ，主要是处理经过Backbone和Neck之后的数据，需要实现以下几个接口:

  * :meth:`~up.tasks.det.models.heads.bbox_head.bbox_head.BboxNet.__init__` 当模块有前驱时，第一个参数为前趋模块输出channel数
  * :meth:`~up.tasks.det.models.heads.bbox_head.bbox_head.BboxNet.forward` input为backbone或者neck的输出，该方法的输出一般为

  .. code-block:: python

   {
     # ... 前面所有模块的输出
     'dt_bboxes': [], # 检测框, RoINet和BboxNet的输出
     'dt_keyps': [], # 检测框对应的keypoints, KeypNet的输出
     'dt_masks': [] # 检测框对应的segmentation, MaskNet的输出
   }

  .. note::

    采用了算法和网络结构分离的设计, 基类(RoINet, BboxNet, KeypNet, MaskNet)实现算法, 子类(NaiveRPN, FC, RFCN, ConvUp)实现具体网络结构

2. 初始化方式和backbone的初始化一致，取决于config参数。
在这个例子中，我们利用前面定义好的CustomNet, 实现一个CustomHead, 完成一个简单的分类网络,新建文件 **custom.py** 放在 `up.tasks.cls.models.heads` 目录下

config 文件示例

  .. code-block:: yaml

    net:
      - name: backbone
         type: custom_net
         kwarg:
           depth: 3
           out_planes: [256]

      - name: head
         prev: backbone
         type: custom_head
         kwarg:
            num_classes: 21


我们使用前面自定义的 :class:`CustomNet` 作为前驱, 设置Head的prev

  .. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

    @MODULE_ZOO_REGISTRY.register('custom_head')
    class CustomHead(nn.Module):
        def __init__(self, in_planes, num_classes):
            """
            由于在配置文件中，我们配置了head有prev部分，因此在构造函数部分会传入in_planes参数                    """

            # build your model..

            self.fc = nn.Linear(inplanes, num_classes)

        def forward(self, input):
            """
            input为字典类型，包含了backbone的输出和dataset的输出
            """

            # implement your algorithm
            # 这里简单使用 global average pooling 和一层 FC

            output = input['features'][0].mean(-1).mean(-1)
            output = self.fc(output)

            loss = self._get_loss(output, input['label'])

            # 将loss放入输出字典中，确保能够让POD对其收集并进行backward
            input['ce_loss'] = loss

        def _get_loss(self, out, label):
            return F.cross_entropy(out, label)

  .. note::

    UP会在最后的输出的字典中寻找所有的包含有loss的项，对他们进行 :meth:`backward` 操作
