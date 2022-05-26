# MSBench Benchmark
## Image Classification Benchmark

### SparseScheduler: AmbaLevelPruneScheduler
| Network/sparsity   | dense | 30%   | 40%   |   50% | 60%   | 70%   |   80% |   90% | 
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|     resnet18       | 70.28 | 70.35 | 70.57 | 70.45 | 70.25 | 69.81 | 68.75 | 63.84 |
|     resnet50       | 76.76 | 76.84 | 76.83 | 76.90 | 76.58 | 76.42 | 75.57 | 72.40 |
| mobilenetv2_0.5    | 65.21 | 64.94 | 64.55 | 63.92 | 62.83 | 60.79 | 56.48 | 46.61 |
| mobilenetv2_1.0    | 73.26 | 73.22 | 72.85 | 72.48 | 71.84 | 69.95 | 67.53 | 54.90 |
| mobilenetv2_2.0    | 77.56 | 77.21 | 77.06 | 76.78 | 76.50 | 75.95 | 74.99 | 68.18 |
| regnetx200         | 68.23 | 68.32 | 68.38 | 67.98 | 67.53 | 66.46 | 64.06 | 58.66 |
| regnetx400         | 71.92 | 72.15 | 72.20 | 71.91 | 71.65 | 70.99 | 69.63 | 64.65 |
| regnety200         | 69.96 | 70.10 | 69.95 | 69.87 | 69.28 | 68.51 | 65.70 | 58.73 |
| regnety400         | 73.41 | 73.51 | 73.55 | 73.45 | 73.08 | 72.43 | 70.81 | 63.74 |


### SparseScheduler: AmpereScheduler
| Network/sparsity   | dense | 50%   |
|--------------------|-------|-------|
| resnet-18          | 70.28 | 70.02 |
| resnet-50          | 76.76 | 76.75 |
| regnety200         | 69.96 | 69.66 |
| regnety400         | 73.41 | 73.71 |



## Object Detection Benchmark

### SparseScheduler: AmbaLevelPruneScheduler
| Network(config)/sparsity                                  | dense |  30%  |  40%  |  50%  |  60%  |  70%  |  80%  |  90%  |
|-----------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| retinanet-FPN(resnet50,bs=32,epoch=7*12)                  | 37.0  |  37.5 | 37.7  | 37.6  | 37.2  |  36.7 |  33.7 |  32.6 |
| retinanet-improve(resnet18,bs=32,epoch=7*12)              | 40.7  |  40.7 |  40.7 | 40.5  |  40.3 | 39.5  |  37.1 |  29.5 |
| retinanet-improve-cos-iou(resnet18,bs=32,epoch=7*12)      | 41.3  |  41.3 |  41.2 |  40.9 |  40.9 | 39.8  |  36.7 |  27.6 |


### SparseScheduler: AmpereScheduler
| Network(config)/sparsity                                  | dense |  50%  |
|-----------------------------------------------------------|-------|-------|
| retinanet-improve(resnet18,bs=32,epoch=7*12)              | 40.7  |  40.3 |
| retinanet-improve-cos-iou(resnet18,bs=32,epoch=7*12)      | 41.3  |  40.8 |

More sparse benchmark can be seen at [sparse benchmark](https://github.com/ModelTC/EOD/tree/main/docs/source/Chinese/benchmark).
