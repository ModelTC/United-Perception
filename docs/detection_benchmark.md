# detection benchmark
| model                   | backbone            | anchor | ema | bs | epoch | general test size | mAP                     |
| ----------------------- | ------------------- | ------ | --- | -- | ----- | ----------------- | ----------------------- |
| retinanet-FPN           | resnet50            |        | no  |  8 | 12    | [800, 1333]       | 37.0                    |
| retinanet-improve       | resnet18            |        | no  | 16 | 100   | [800, 1333]       | 40.7                    |
| faster-rcnn-FPN         | resnet50            |        | no  |  8 | 12    | [800, 1333]       | 38.2                    |
| faster-rcnn-C4          | resnet50            |        | no  |  8 | 12    | [800, 1333]       |                         |
| YOLOX                   | nano                |        | yes | 16 | 300   | [416, 416]        | 24.8                    |
| YOLOX                   | tiny                |        | yes | 16 | 300   | [416, 416]        | 33.0                    |
| YOLOX                   | small               |        | yes | 16 | 300   | [640, 640]        | 40.4                    |
| YOLOX                   | medium              |        | yes | 16 | 300   | [640, 640]        | 46.9                    |
| YOLOX                   | large               |        | yes | 16 | 300   | [640, 640]        | 49.9                    |
| YOLOv5-small            | darknetv5           |        | yes | 16 | 300   | [640, 640]        | 37.4                    |
| YOLOX-ret               | nano                |   1    | yes | 16 | 300   | [416, 416]        | 25.8                    |
| YOLOX-ret               | nano                |   2    | yes | 16 | 300   | [416, 416]        | 26.4                    |
| YOLOX-ret-ada           | nano                |   2    | yes | 16 | 300   | [416, 416]        | 27.2                    |
| YOLOX-ret               | tiny                |   1    | yes | 16 | 300   | [416, 416]        | 33.6                    |
| YOLOX-ret               | tiny                |   2    | yes | 16 | 300   | [416, 416]        | 33.8                    |
| YOLOX-ret-ada           | tiny                |   2    | yes | 16 | 300   | [416, 416]        | 35.4                    |
| YOLOX-ret               | small               |   1    | yes | 16 | 300   | [640, 640]        | 40.4                    |
| YOLOX-ret               | small               |   2    | yes | 16 | 300   | [640, 640]        | 40.7                    |
| YOLOX-ret               | medium              |   1    | yes | 16 | 300   | [640, 640]        | 47.0                    |
| YOLOX-ret               | medium              |   2    | yes | 16 | 300   | [640, 640]        | 47.4                    |
