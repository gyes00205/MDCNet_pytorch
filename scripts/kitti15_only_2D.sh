python finetune_only_one_network.py \
    --model Unet --train_bsize 4 --test_bsize 1 \
    --save_path ablation_study/kitti15_only_2D \
    --pretrained ablation_study/pretrained_only_2D_network/checkpoint23.tar \
    --split_file dataset/KITTI2015_val.txt