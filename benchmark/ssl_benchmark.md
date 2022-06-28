## MocoV1

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [resnet50](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mocov1/moco_v1.yaml) | Steplr  | 224        | 512  | 200   | [download](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MocoV1_pretrain.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [resnet50](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mocov1/moco_v1_imagenet_linear.yaml) | Steplr  | 224        | 256  | 100   | [60.75](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MocoV1_linear_finetune.pth) |



## MocoV2

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [resnet50](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mocov2/moco_v2.yaml) | coslr   | 224        | 512  | 200   | [download](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MocoV2_pretrain.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [resnet50](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mocov2/moco_v2_imagenet_linear.yaml) | Steplr  | 224        | 256  | 100   | [67.11](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MocoV2_linear_finetune.pth) |



## SimSiam

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [resnet50](https://github.com/ModelTC/EOD/tree/main/configs/ssl/simsiam/simsiam_100e.yaml) | coslr   | 224        | 512  | 100   | [download](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/SimSiam_pretrain.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [resnet50](https://github.com/ModelTC/EOD/tree/main/configs/ssl/simsiam/simsiam_linear.yaml) | coslr   | 224        | 256  | 90    | [68.02](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/SimSiam_linear_finetune.pth) |



## MAE

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [vit-b16](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mae/mae_vit_base_patch16_dec512d8b_800e.yaml) | coslr   | 224        | 4096 | 800   | [download](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MAE_pretrain.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [vit-b16](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mae/mae_vit_base_patch16_dec512d8b_linear.yaml) | coslr   | 224        | 8192 | 100   | [66.80](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MAE_linear_finetune.pth) |

- full finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [vit-b16](https://github.com/ModelTC/EOD/tree/main/configs/ssl/mae/mae_vit_base_patch16_dec512d8b_finetune.yaml) | coslr   | 224        | 1024 | 100   | [83.12](https://github.com/ModelTC/United-Perception/releases/download/0.2.0_github/MAE_full_finetune.pth) |

