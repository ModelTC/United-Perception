# EOD Developing Mode
We highly recommand a novel developing mode called **Public EOD + Plugins**, which composed by two components:

* Public EOD: Completed detection frame.
* Plugins: Customized modules registered by REGISTRYs.

## Customized Plugin
You can develop a plugin which organized by several registered modules, such as datasets, models, losses, etc. Take **Face** package for instance, the structure is as follows: 

```
face
├── datasets
|   └── face_dataset.py
├── __init__.py
└── models
    ├── facenet.py
    └── __init__.py
```
FaceDataset and FaceNet defined in package need to be registered with DATASET_REGISTRY and MODULE_ZOO_REGISTRY.

Then you need to add package into PLUGINPATH:

```shell
export PLUGINPATH='path to father_dir_of_face'
```

## EOD Development Mode
Design described above have following advantages:

* Import flexibly: After developing a plugin, you only need to add path into PLUGINPATH.
* Use conveniently: You could origanize pipeline by only adding register aliases into configs. Register details refer to [Register](register_modules.md).
* Maintenance friendly: Public EOD is well isolated from personal plugins, you only need to maintain your plugins code with few costs.
