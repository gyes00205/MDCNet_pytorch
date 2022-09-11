python finetune_only_one_network.py \
    --model stackhourglass --train_bsize 2 --test_bsize 1 --datatype 2012 \
    --datapath /home/bsplab/Documents/data_stereo_flow_2012/training/ \
    --save_path ablation_study/kitti12_only_3D \
    --pretrained ablation_study/pretrained_only_3D_network/checkpoint15.tar \
    --split_file dataset/KITTI2012_val.txt