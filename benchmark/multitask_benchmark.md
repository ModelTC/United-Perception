# multitask benchmark (det baseline: r50-retina-atss-qfl)
| task                   | backbone            | AP    | top1  | model |
| ---------------------- | ------------------- | ----- | ----- | ----- |
| [det](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/multitask/r50-retina-atss-qfl.yaml)                    | resnet50            | 39.20 | -     | [ckpt](http://spring.sensetime.com/dropadmin/$/FzCjp.pth) |
| [det + cls](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/multitask/r50-retina-atss-qfl+cls.yaml)              | resnet50            | 40.01 | 67.81 | [ckpt](http://spring.sensetime.com/dropadmin/$/mRoOR.pth) |
