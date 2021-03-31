#!/usr/bin/env bash

############################### 参数意义 ################################
# CONFIG:  配置文件的路径，列出的四个都可以用
# display_port: 端口号
# device： 显卡号 如果用 cpu训练或者测试 就写 cpu
# which_epoch： 加载预训练模型训练，这个模型要存在于 wordir 的目录中
######################################################################### 

PYTHON=${PYTHON:-"python"}
# CONFIG="./configs/restore_train.yaml"           ## 常规mask损失
# CONFIG="./configs/restore_mix_train.yaml"       ## 对抗损失 + 常规损失
# CONFIG="./configs/restore_gan_train.yaml"       ## gan的方法
CONFIG="./configs/restore_fusion_train.yaml"      ## 多阶段融合的方法

#### resume 单卡训练  8039  48135
$PYTHON train.py --config $CONFIG  --display_port=6006 --device cuda:0 --which_epoch stage12
# $PYTHON train.py --config $CONFIG  --display_port=8039 --device cuda:0 

