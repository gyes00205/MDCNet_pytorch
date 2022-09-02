python finetune_only_one_network.py \
    --model stackhourglass --train_bsize 2 --test_bsize 1 \
    --save_path ablation_study/kitti15_only_3D \
    --pretrained ablation_study/pretrained_only_3D_network/checkpoint15.tar \
    --split_file dataset/KITTI2015_val.txt