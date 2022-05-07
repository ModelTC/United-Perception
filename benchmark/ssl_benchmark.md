## MocoV1

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mocov1/moco_v1.yaml) | Steplr  | 224        | 512  | 200   | [download](http://spring.sensetime.com/dropadmin/$/vktKI.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mocov1/moco_v1_imagenet_linear.yaml) | Steplr  | 224        | 256  | 100   | [60.75](http://spring.sensetime.com/dropadmin/$/o5YBo.pth) |



## MocoV2

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mocov2/moco_v2.yaml) | coslr   | 224        | 512  | 200   | [download](http://spring.sensetime.com/dropadmin/$/wgUnK.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mocov2/moco_v2_imagenet_linear.yaml) | Steplr  | 224        | 256  | 100   | [67.11](http://spring.sensetime.com/dropadmin/$/yDFNy.pth) |



## SimSiam

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/simsiam/simsiam_100e.yaml) | coslr   | 224        | 512  | 100   | [download](http://spring.sensetime.com/dropadmin/$/FmhxD.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [resnet50](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/simsiam/simsiam_linear.yaml) | coslr   | 224        | 256  | 90    | [68.02](http://spring.sensetime.com/dropadmin/$/L7BJG.pth) |



## MAE

- pretrain

| model                                                        | setting | train size | bs   | epoch | model                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ------------------------------------------------------------ |
| [vit-b16](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mae/mae_vit_base_patch16_dec512d8b_800e.yaml) | coslr   | 224        | 4096 | 800   | [download](http://spring.sensetime.com/dropadmin/$/6UTFT.pth) |

- linear finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [vit-b16](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mae/mae_vit_base_patch16_dec512d8b_linear.yaml) | coslr   | 224        | 8192 | 100   | [66.80](http://spring.sensetime.com/dropadmin/$/j7dKF.pth) |

- full finetune 

| model                                                        | setting | train size | bs   | epoch | acc                                                        |
| ------------------------------------------------------------ | ------- | ---------- | ---- | ----- | ---------------------------------------------------------- |
| [vit-b16](https://gitlab.bj.sensetime.com/spring2/united-perception/-/blob/dev/configs/ssl/mae/mae_vit_base_patch16_dec512d8b_finetune.yaml) | coslr   | 224        | 1024 | 100   | [83.12](http://spring.sensetime.com/dropadmin/$/HDBpO.pth) |

