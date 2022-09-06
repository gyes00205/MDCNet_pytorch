# MDCNet
###### tags: `MDCNet`
## Introduction
Thanks for the amazing work of Wei Chen, Xiaogang Jia, Mingfei Wu, and Zhengfa Liang. The goal of this repo is to implement and reproduce the paper **Multi-Dimensional Cooperative Network for Stereo Matching** which published on ICRA 2022. Original paper could be found via the following links:
* [Original paper](https://ieeexplore.ieee.org/document/9627805)

## Dataset
* [SceneFlow](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
* [KITTI Stereo](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

I follow the data preparation of [PSMNet](https://github.com/JiaRenChang/PSMNet/tree/master/dataset) to download the SceneFlow dataset. 
## Ablation Study
Train on RTX 2080Ti

| Matching Cost Computation | Cost Aggregation         | SceneFlow (EPE) | KITTI 2015 D1-all (%) | KITTI 2012 D1-all (%) | Time(s) |
| ------------------------- | ------------------------ | --------------- | --------------------- | --------------------- | ------- |
| Correlation               | Unet/2D                  | 1.647           | 3.93%                 |                       | 0.043   |
| Concat                    | Hourglass/3D             | 1.417           | 2.13%                 |                       | 0.243   |
| Correlation+Concat        | Unet/2D+DCU+Hourglass/3D | 1.351           | 3.29%                 |                       | 0.073   |

## Acknowledgements
In this implementation, I use parts of the implementations of the following works:
* [PSMNet](https://github.com/JiaRenChang/PSMNet) by [Jia-Ren Chang](https://jiarenchang.github.io/)
* [GWCNet](https://github.com/xy-guo/GwcNet) by [Xiaoyang Guo](https://github.com/xy-guo)
* [CasStereoNet](https://github.com/hz-ants/cascade-mvsnet) by [Xiaodong Gu](https://github.com/gxd1994)
* [AnyNet](https://github.com/mileyan/AnyNet) by [Yan Wang](https://www.cs.cornell.edu/~yanwang/)

Thanks for the respective authors for sharing their amazing works.