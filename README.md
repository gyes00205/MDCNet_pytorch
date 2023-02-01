# MDCNet
###### tags: `paper re-implementation`
## :beginner: Introduction
![demo.gif animation](readme_images/demo.gif)

Thanks for the amazing work of Wei Chen, Xiaogang Jia, Mingfei Wu, and Zhengfa Liang. The goal of this repo is to implement and reproduce the paper **Multi-Dimensional Cooperative Network for Stereo Matching** which published on ICRA 2022. Original paper could be found via the following links:
* [Original paper](https://ieeexplore.ieee.org/document/9627805)
* [Original repo](https://github.com/disco14/MDCNet)

## :floppy_disk: Dataset
* [SceneFlow](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
* [KITTI Stereo](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

I follow the data preparation of [PSMNet](https://github.com/JiaRenChang/PSMNet/tree/master/dataset) to download the SceneFlow dataset.
* SceneFlow includes three datasets: flything3d, driving and monkaa.
* You can train MDCNet with some of three datasets, or all of them.
* the following is the describtion of six subfolder.
```
# the disp folder of Driving dataset
driving_disparity  
# the image folder of Driving dataset
driving_frames_cleanpass

# the disp folder of  Flything3D dataset
frames_cleanpass  
# the image folder of  Flything3D dataset
frames_disparity  

# the disp folder of Monkaa dataset
monkaa_disparity  
# the image folder of Monkaa dataset
monkaa_frames_cleanpass
```

## :hourglass: Training
### Pretrain on SceneFlow dataset
I pretrain MDCNet on SceneFlow dataset for 24 epochs.
```
bash scripts/sceneflow_mdcnet.sh
```
### Finetune on KITTI 2015
I finetune MDCNet on KITTI 2015 dataset for 300 epochs. Split 80% data for training and 20% for validation.
```
bash scripts/kitti15_mdcnet.sh
```
### Finetune on KITTI 2012
I finetune MDCNet on KITTI 2012 dataset for 300 epochs. Split 80% data for training and 20% for validation.
```
bash scripts/kitti12_mdcnet.sh
```

## :rocket: Inference
You can inference on kitti raw dataset.
```
python test_loop.py \
    --datapath /home/bsplab/Documents/dataset_kitti/train/2011_09_26_drive_0011_sync \
    --output_dir output \
    --loadmodel results/kitti15_mdcnet/checkpoint.tar
```

## :chart_with_upwards_trend: Ablation Study
Train on RTX 2080Ti

| Matching Cost Computation | Cost Aggregation         | SceneFlow (EPE) | KITTI 2015 D1-all (%) | KITTI 2012 D1-all (%) | Time(s) |
| ------------------------- | ------------------------ | --------------- | --------------------- | --------------------- | ------- |
| Correlation               | Unet/2D                  | 1.647           | 3.93%                 | 5.08%                 | 0.043   |
| Concat                    | Hourglass/3D             | 1.121           | 2.13%                 | 2.56%                 | 0.243   |
| Correlation+Concat        | Unet/2D+DCU+Hourglass/3D | 1.351           | 3.16%                 | 3.91%                 | 0.073   |

## :two_hearts: Acknowledgements
In this implementation, I use parts of the implementations of the following works:
* [PSMNet](https://github.com/JiaRenChang/PSMNet) by [Jia-Ren Chang](https://jiarenchang.github.io/)
* [GWCNet](https://github.com/xy-guo/GwcNet) by [Xiaoyang Guo](https://github.com/xy-guo)
* [CasStereoNet](https://github.com/hz-ants/cascade-mvsnet) by [Xiaodong Gu](https://github.com/gxd1994)
* [AnyNet](https://github.com/mileyan/AnyNet) by [Yan Wang](https://www.cs.cornell.edu/~yanwang/)

Thanks for the respective authors for sharing their amazing works.