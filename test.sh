#!/usr/bin/env bash
############################### 参数意义 ################################
# CONFIG:  配置文件的路径，和训练的时候使用的配置文件路径不同，测试的时候使用
#          checkpoints中输出的配置文件，模型也存在checkpoints文件夹中
# DATA_DIR： 测试数据的路径
# DATASETMODE： 只能设置为unaligned
# device： 显卡号 如果用 cpu训练或者测试 就写 cpu
# which_epoch： 模型名字 latest_net_G_A.pth 则 which_epoch = latest
######################################################################### 
PYTHON=${PYTHON:-"python"}

### 以下都是训练过的模型，可以直接用来测试
# CONFIG="./checkpoints/restore_qc_mask_loss/opt.yaml"
# CONFIG="./checkpoints/restore_qc_deconv/opt.yaml"
# CONFIG="./checkpoints/restore_qc_mask_atten/opt.yaml"

# CONFIG="./checkpoints/restore_gan_mse/opt.yaml"
# CONFIG="./checkpoints/restore_ragan/opt.yaml"
# CONFIG="./checkpoints/restore_unetgan/opt.yaml"

# CONFIG="./checkpoints/restore_spa_exp/opt.yaml"
# CONFIG="./checkpoints/restore_gan_spa_exp_no_att/opt.yaml"
# CONFIG="./checkpoints/restore_cur_trans/opt.yaml"

# CONFIG="./checkpoints/restore_fusion_pre_mask/opt.yaml"
CONFIG="./checkpoints/restore_fusion_gan/opt.yaml"


DATA_DIR='/home/arc-crw5713/data/low_light_test_qc_-2ev'
DATASETMODE='unaligned'

# #### predict 
$PYTHON predict.py --config $CONFIG --resize_or_crop='no' --dataroot $DATA_DIR --device cuda:0\
                   --dataset_mode $DATASETMODE --which_epoch latest
 