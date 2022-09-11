python finetune_only_one_network.py \
    --model Unet --train_bsize 4 --test_bsize 1 --datatype 2012 \
    --datapath /home/bsplab/Documents/data_stereo_flow_2012/training/ \
    --save_path ablation_study/kitti12_only_2D \
    --pretrained ablation_study/pretrained_only_2D_network/checkpoint23.tar \
    --split_file dataset/KITTI2012_val.txt