# Register Modules
EOD supports register for each module to flexibly compose pipelines.

## How to register
All callable object(inclue function, classes, .etc) can be registered, but the result should be corresponding instance. For example. CocoDataset return a dataset, and  resnet50 return a module(ResNet instance).
The registration format is: REGISTRY.register(alias).

Registry is corresponding instance, for example, DATASET_REGISTRY, AUGMENTATION_REGISTRY, etc.

The following are examples of registering CocoDataset and resnet50 respectively:

```python
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
```

## How to use Registry
EOD calls registered modules according to config. Alias should be provided for locating the module.
For example, CocoDataset could be called by type "coco"(alias) and parameters "kwargs".

```yaml
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
```

