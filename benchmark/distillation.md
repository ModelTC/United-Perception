# Distillation

## Detection

Results on COCO dataset. Teacher and Student performance:

|config  | scheduler | AP | AP50 | AP75 | APs | APm | APl |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Student(Res50) | 12e | 38.0 | 59.1 | 41.1 | 23.0 | 41.9 | 48.3 |
| Teacher(Res152) | 12e | 42.3 | 63.4 | 46.2 | 26.3 | 46.5 | 54.0 |

Mimic methods performance:

|config  | scheduler | AP | AP50 | AP75 | APs | APm | APl |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [baseline(neck_mimic)](https://github.com/ModelTC/EOD/blob/main/configs/distiller/det/faster_rcnn/faster_rcnn_r152_50_1x_feature_mimic.yaml) | 12e | 39.5 | 60.5 | 43.2 | 24.2 | 42.9 | 51.5 |
| [SampleFeature](https://github.com/ModelTC/EOD/blob/main/configs/distiller/det/faster_rcnn/faster_rcnn_r152_50_1x_sample_feature_mimic.yaml) | 12e | 39.8 | 60.7 | 43.2 | 24.1 | 43.5 | 50.7 |
| [FRS](https://github.com/ModelTC/EOD/blob/main/configs/distiller/det/faster_rcnn/faster_rcnn_r152_50_1x_frs.yaml) | 12e | 40.9 | 61.5 | 44.4 | 24.5 | 45.0 | 52.7 |
| [DeFeat](https://github.com/ModelTC/EOD/blob/main/configs/distiller/det/faster_rcnn/faster_rcnn_r152_50_1x_decouple_feature_mimic.yaml) | 12e | 41.0 | 61.8 | 44.8 | 24.0 | 45.4 | 53.3 |

An advanced usage of the distillation is to mimic one student with multi teachers by multi methods like [link](https://github.com/ModelTC/EOD/blob/main/configs/distiller/det/faster_rcnn/faster_rcnn_r152_50_1x_multi_jobs_multi_teacheres.yaml)

