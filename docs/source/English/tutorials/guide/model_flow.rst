Model flow
==========

* :ref:`BackboneNeckAnchorEn`
* :ref:`HeadAnchorEn`

Models are built by multiple sub-modules. The sub-modules are independent with each other and the communications between them are realized by the set inferfaces.
The type of inputs and outputs between modules is 'dictionary'.

Every model structure can be abstracted as 'feature extracting' + 'task branch', where 'feature extracting' contains backbone and neck, and 'task branch' has different structures according to the tasks.

.. _BackboneNeckAnchorEn:

**BackBone & Neck**
~~~~~~~~~~~~~~~~~~~

1. All Backbones and Necks inherit from class:`torch.nn.Module`, and need to realize the following interfaces.

  * :meth:`~up.models.backbones.ResNet.__init__` When former network exists, the first parameter is the output channel number of it.
  * :meth:`~up.models.backbones.ResNet.get_outplanes` When latter network exists, the function should be realized to return the output channel number for constructing the latter network.
  * :meth:`~up.models.backbones.ResNet.forward` The input is the output of the dataset. The output format of the method is as followed.

  .. code-block:: python

    {'features':[], 'strides': []}

2. :meth:`__init__` The parameters of function is mainly depended on configs. For example, we firstly use an simplest config `custom_net.yaml` to build a CNN with L layers.

  .. note::

    Ensure the defined type in configs to be imported. In this instance, we create a file 'custom.py' under 'up.models.backbones'.
    You can use registry to register a new module and give it an alias, and then call the module by its alias.

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
            Structural parameters come from the kwargs in the config.
            The instance has no inplances parameter since it has no precursor module.
            """
            self.out_planes = out_planes
          
            in_planes = 3
            for i in range(depth):
                self.add_module(f'layer{i}',
                                nn.Conv2d(in_planes, out_planes[i], kernel_size=3, padding=1))
                self.add_module('relu', nn.ReLU(inplace=True))
                in_planes = out_planes[i]

Then we realize :meth:`forward` and :meth:`get_outplanes`

  .. note::

    :meth:`foward` function needs computing the output features and strides that are both array format.

  .. code-block:: python

    def forward(self, input):
        """
        The type of input (dictionary) and the organization of data are mainly decided by Dataset in config.
        Here we assume that the input contains images.
        """

        x = input['image']

        for submodule in self.children():
            x = submodule(x)

        # The output is a dictionary which must contain features and strides, in the meanwhile we keep other data in the input.
        input['features'] = [x]
        input['strides'] = [1]

        return input

    def get_outplanes(self):

        return self.out_planes

  .. note::
   
    The backbone can be called in '__init__.py', and will be automatically registered to 'MODULE_ZOO_REGISTRY'.
    The neck for detection and segmentation needs being registered to 'MODULE_ZOO_REGISTRY'by '@MODULE_ZOO_REGISTRY.register("bias")'.

.. _HeadAnchorEn:

**Head**
~~~~~~~~

1. Head module inherits class:`torch.nn.Module`, and mainly tackles the data output from Backbone and Neck. It needs the following interfaces.

  * :meth:`~up.tasks.det.models.heads.bbox_head.bbox_head.BboxNet.__init__` The first parameter is the output channel of the precursor if the precursor exist.
  * :meth:`~up.tasks.det.models.heads.bbox_head.bbox_head.BboxNet.forward` The input of it comes from the output of backbones or necks. The output of it is as followed.

  .. code-block:: python

   {
     # ... all outputs of the previous modules.
     'dt_bboxes': [], # Detection boxes from RoINet and BboxNet.
     'dt_keyps': [], # Detection keypoints from KeypNet.
     'dt_masks': [] # Segmentation from MaskNet which comes from the detection boxes.
   }

  .. note::

    The algorithm and network are seperately designed. Base classes including RoINet, BboxNet, KeypNet, and MaskNet realize the algorithm while sub-classes including NaiveRPN, FC, RFCN, and ConvUp realize the specific network structure.

2. The initialization way is the same as that for backbones, which is decided by parameters in the config.
In this instance, we use the previously defined 'CustomNet' to realize a 'CustomHead' and build a simple classification network. The created **custom.py** is under `up.tasks.cls.models.heads` .

config is as followed.

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


we use the previously defined class:`CustomNet` as the precursor to set the prev of Head.

  .. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY

    @MODULE_ZOO_REGISTRY.register('custom_head')
    class CustomHead(nn.Module):
        def __init__(self, in_planes, num_classes):
            """
            Since we have set the prev of Head in the config, the constructor will input in_planes.
            """
            # build your model..

            self.fc = nn.Linear(inplanes, num_classes)

        def forward(self, input):
            """
            input is dictionary containing the output of backbone and dataset.
            """

            # implement your algorithm
            # Simply use global average pooling and a FC layer.

            output = input['features'][0].mean(-1).mean(-1)
            output = self.fc(output)

            loss = self._get_loss(output, input['label'])

            # Add loss into the dictionary to ensure that UP can get it for backwarding.
            input['ce_loss'] = loss

        def _get_loss(self, out, label):
            return F.cross_entropy(out, label)

  .. note::
    
    UP will find all items containing any loss in the output dictionary and execute :meth:`backward` on them.
