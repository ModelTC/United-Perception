# Learn About Configs

EOD sets pipeline parameters by incorporating them into configs. 
Considering convenience and expansibility, EOD offers common interfaces for all components such as dataset, saver, backbone, etc. which are implemented by REGISTRY [Register](register_modules.md).

## Configuration File
Standard information includes num_classes, runtime, dataset, trainer, saver, hooks and net. The standard format is as follows:
```yaml
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
```

## Details
Each item corresponds to a registered entity.

```yaml
crop: &crop
  type: crop
  kwargs:
    means: [123.675, 116.280, 103.530]
    scale: 1024
    crop_prob: 0.5
```
