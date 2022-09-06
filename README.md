# MDCNet
###### tags: `MDCNet`
## Introduction
The goal of this repo is to implement and reproduce the paper **Multi-Dimensional Cooperative Network for Stereo Matching** which published on ICRA 2022. Original paper could be found via the following links:
* [Original paper](https://ieeexplore.ieee.org/document/9627805)

## Ablation Study
Result on SceneFlow


| Matching Cost Computation | Cost Aggregation | SceneFlow (EPE) | KITTI 2015 | KITTI 2012 | Time(s) |
| ------------------------- | ---------------- | --------------- | ---------- | ---------- | ------- |
| Correlation               | Unet/2D          | 1.647           | 3.93%      |            |         |
| Concat                    | Hourglass/3D     | 1.417           | 2.13%      |            |         |
|                           |                  |                 |            |            |         |
|                           |                  |                 |            |            |         |

## Acknowledgements
In this implementation, I use parts of the implementations of the following works:
* [PSMNet](https://github.com/JiaRenChang/PSMNet) by [Jia-Ren Chang](https://jiarenchang.github.io/)
* [GWCNet](https://github.com/xy-guo/GwcNet) by [Xiaoyang Guo](https://github.com/xy-guo)
* [CasStereoNet](https://github.com/hz-ants/cascade-mvsnet) by [Xiaodong Gu](https://github.com/gxd1994)
* [AnyNet](https://github.com/mileyan/AnyNet) by [Yan Wang](https://www.cs.cornell.edu/~yanwang/)

Thanks for the respective authors for sharing their amazing works.