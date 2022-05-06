# quant benchmark 

Based on Mqbench and UP , we provide an object detection benchmark on COCO dataset.

| Model                 | Backend   | map\@fp32 | ptq\@int8 | qat\@int8 |
|-----------------------|-----------|----------|-----------|----------|
| retinanet-r18-improve | tensorrt  | 40.7      | 40.5     | 40.7     |
| retinanet-r18-improve | snpe      | 40.7      | 39.7     | 40.2     |
| retinanet-r18-improve | vitis     | 40.7      | 39.0     | 40.1     |
| yolox_s               | tensorrt  | 40.5      | 39.4     | 39.8     |
| yolox_s               | snpe      | 40.5      | 38.1     | 39.8     |
| yolox_s_lpcv          | vitis     | 29.3      | 25.3     | 27.4     |
| faster-rcnn-improve   | tensorrt  | 43.6      | 43.1     | 44.8     |
