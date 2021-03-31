#!/usr/bin/env bash

############################### 参数意义 ################################
# CONFIG:  配置文件的路径，列出的l两个都可以用 ohem 训练 但是restore_fusion_ohem_train
#          训练没效果
# display_port: 端口号
# device： 显卡号 如果用 cpu训练或者测试 就写 cpu
# which_epoch： 加载预训练模型训练，这个模型要存在于 wordir 的目录中
######################################################################### 

## ohem 使用了单独的配置文件，主要是修改了 学习率参数，预加载模型参数，损失权重

PYTHON=${PYTHON:-"python"}
CONFIG="./configs/restore_ohem_train.yaml"
# CONFIG="./configs/restore_fusion_ohem_train.yaml"


#### resume 单卡训练  35725  16127
# $PYTHON train.py --config $CONFIG  --display_port=35725 --device cuda:0 --continue_train --which_epoch latest
$PYTHON train_ohem.py --config $CONFIG  --display_port=35725 --device cuda:0

