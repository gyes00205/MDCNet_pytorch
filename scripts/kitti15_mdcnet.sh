python finetune_only_one_network.py \
    --train_bsize 4 --test_bsize 1 --datatype 2015 \
    --datapath /home/bsplab/Documents/data_scene_flow_2015/training/ \
    --save_path results/kitti15_mdcnet \
    --pretrained results/pretrained_mdcnet/checkpoint23.tar \
    --split_file dataset/KITTI2015_val.txt