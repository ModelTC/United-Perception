# semantic benchmark
| model                   | backbone            | ema | crop test | bs  | epoch | general test size | mIoU                           |
| ----------------------- | ------------------- | --- | --------- | --- | ----- | ----------------- | ------------------------------ |
| deeplabv3 (light-seg)   | resnet50            | no  | -         | 8   | 120   | [2048, 1024]      | 79.5                           |
| deeplabv3               | resnet50            | no  | -         | 8   | 120   | [2048, 1024]      | 79.5                           |
| deeplabv3               | resnet50            | yes | -         | 8   | 120   | [2048, 1024]      | 80.5 \| 80.2                   |
| deeplabv3 (light-seg)   | resnet101           | no  | -         | 8   | 120   | [2048, 1024]      | 80.8                           |
| deeplabv3               | resnet101           | no  | -         | 8   | 120   | [2048, 1024]      | 81.2                           |
| deeplabv3               | resnet101           | yes | -         | 8   | 120   | [2048, 1024]      | 81.8                           |
| df_seg_v1  (light-seg)  | dfnet1              | no  | -         | 16  | 180   | [2048, 1024]      | 70.62                          |
| df_seg_v1               | dfnet1              | no  | -         | 16  | 180   | [2048, 1024]      | 71.8 \| 71.7                   |
| df_seg_v1 mseg pretrain | dfnet1              | no  | -         | 16  | 180   | [2048, 1024]      | 72.8                           |
| df_seg_v1               | dfnet1              | yes | -         | 16  | 180   | [2048, 1024]      | 71.6 \| 71.7                   |
| df_seg_v2  (light-seg)  | dfnet2              | no  | -         | 16  | 180   | [2048, 1024]      | 75.44                          |
| df_seg_v2               | dfnet2              | no  | -         | 16  | 180   | [2048, 1024]      | 74.6 \| 75.0                   |
| df_seg_v2               | dfnet2              | yes | -         | 16  | 180   | [2048, 1024]      | 76.2 \| 75.8                   |
| sf_seg_v1  (light-seg)  | dfnet1              | no  | -         | 16  | 180   | [2048, 1024]      | 72.11                          |
| sf_seg_v1               | dfnet1              | no  | -         | 16  | 180   | [2048, 1024]      | 72.5 (light-seg 72.11)         |
| sf_seg_v1               | dfnet1              | yes | -         | 16  | 180   | [2048, 1024]      | 72.96                          |
| sf_seg_v2  light-seg    | dfnet2              | no  | -         | 16  | 180   | [2048, 1024]      | 76.5                           |
| sf_seg_v2               | dfnet2              | no  | -         | 16  | 180   | [2048, 1024]      | 76.5                           |
| sf_seg_v2               | dfnet2              | yes | -         | 16  | 180   | [2048, 1024]      | 77.4                           |
| pspnet    light-seg     | res50               | no  | -         | 8   | 120   | [2048, 1024]      | 75.93                          |
| pspnet                  | res50               | no  | -         | 8   | 120   | [2048, 1024]      | 77.46                          |
| pspnet                  | res50               | yes | -         | 8   | 120   | [2048, 1024]      | 77.9                           |
| BiSegNet  light-seg     | res18 (deep pt)     | no  | -         | 16  | 300   | [2048, 1024]      | 75.0                           |
| BiSegNet                | res18 (deep pt)     | no  | -         | 16  | 300   | [2048, 1024]      | 75.8 (light-seg 75.0)          |
| BiSegNet                | res18 (deep pt)     | yes | -         | 16  | 300   | [2048, 1024]      | 76.4                           |
| ICNet   light-seg       | res50               | no  | -         | 16  | 120   | [2048, 1024]      | 75.62                          |
| ICNet                   | res50               | no  | -         | 16  | 120   | [2048, 1024]      | 76.0                           |
| ICNet                   | res50               | yes | -         | 16  | 120   | [2048, 1024]      | 76.8                           |
| UNet (769) light-seg    | unet                | no  | -         | 8   | 120   | [2048, 1024]      | 53.06                          |
| UNet (769)              | unet                | no  | -         | 8   | 120   | [2048, 1024]      | 64.4                           |
| UNet (769)              | unet                | yes | -         | 8   | 120   | [2048, 1024]      | 63.0                           |
| UNet (1024) light-seg   | unet                | no  | -         | 8   | 240   | [2048, 1024]      | 53.37                          |
| UNet (1024)             | unet                | no  | -         | 8   | 240   | [2048, 1024]      | 67.4                           |
| UNet (1024)             | unet                | yes | -         | 8   | 240   | [2048, 1024]      | 68.0                           |



