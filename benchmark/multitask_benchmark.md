# multitask benchmark (det baseline: r50-retina-atss-qfl)
| task                   | backbone            | AP    | top1  | model |
| ---------------------- | ------------------- | ----- | ----- | ----- |
| [det](https://github.com/ModelTC/United-Perception/tree/main/configs/multitask/r50-retina-atss-qfl.yaml)                    | resnet50            | 39.20 | -     | [ckpt](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/det.pth) |
| [det + cls](https://github.com/ModelTC/United-Perception/tree/main/configs/multitask/r50-retina-atss-qfl+cls.yaml)              | resnet50            | 40.01 | 67.81 | [ckpt](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/det+cls.pth) |
