python finetune.py \
    --train_bsize 4 --test_bsize 1 --datatype 2012 \
    --datapath /home/bsplab/Documents/data_stereo_flow_2012/training/ \
    --save_path results/kitti12_mdcnet \
    --pretrained results/pretrained_mdcnet/checkpoint23.tar \
    --split_file dataset/KITTI2012_val.txt