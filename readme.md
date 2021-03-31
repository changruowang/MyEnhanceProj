# 工程说明
提亮工程中尝试过的，三类方法，分别由/model文件夹中的 restore_model.py, restore_gan_model.py 和 restore_fusion_model.py 实现，对应的配置文件为 restore_train.yaml, restore_gan_train.yaml 和 restore_fusion_train.yaml，这三个为实现对应方法所需要的base基本参数配置。其他命名的配置文件为每个base方法的衍生测试。

# yaml 参数解释  
由于方法是时间顺序编写的，每种方法使用的参数有交集，但是并不是配置文件中的参数在每个方法中都有用，因此如果要使用新增或者修改基本配置文件中的参数，最好在代码中确认有效。

## 数据集相关的参数   
* train_dataroot： 包含.h5的数据集文件路径  
* test_dateroot: 包含 testA,testB 文件夹的测试图象路径 testA文件夹里放低光图
* dataset_mode: 选择使用的数据集类型
    * hd5f: 训练的时候使用，加载预先制作好的 .h5 数据集
    * unaligned: 测试的时候用，用于直接读图
* mask_type：
    * poly: 使用数据集中预先存在的 mask 
    * rect： 使用同路径下，标注的前景运动框 .json 来生成矩形的 mask 
* random_mask：数据集最终输出的 mask （键为'motion_mask'） 的形状 
    * True: random画刷随机生成的mask
    * False: 输出 mask 由 mask_type 参数决定的形态
* resize_or_crop： 测试的时候为 'no' 训练的时候不为 'no' 即可
* aligen_AB： 输出的 AB 图像是否是同一场景 
    * False： 真实样本 B（键为'A_input_ref'） 就是从所有正常图象中随便选
    * True: 等同于A的参考图 
* box_filter： True，则将输出的 'A_input_ref' 模糊，否则就等于A_ref，A_input_ref 一般用于和 低光图 cat输入网络 
* fineSize： 最终的 patch_size 大小（ 320 ）
* motion_sample_percent： crop 运动区域的概率 0-1，如果取0，则训练样本完全是从对齐的区域中crop 

## 生成模型相关参数
* name：workdir 训练中间结果保存的文件夹名
* model: 选择提亮的主体方法，目前支持三种。
    * restore_model：常规的损失函数，端到端训练，调用restore_model.py中的模型 
    * restore_gan_model: 基于gan方法的模型，整合了所有尝试过的gan方法
    * fusion_model: 基于多阶段的 融合模型，调用restore_fusion_model.py中的模型
* which_model_netG: 选择生成网络的类型，详细可选主干网络类型见代码注释
* output_mode：生成网络输出模式
    * direct_img：网络直接输出增强后的图象
    * cur_trans： 网络输出曲线变换参数 
    * line_trans：网络输出线性变换参数
* output_nc： 生成网络的输出通道数（主要是用于输出线性变换这类方法使用）
* tanh：生成网络输出是否 tanh 激活 默认True即可
* skip：生成网络输出使用跨层连接，即预测输入图的残差
* which_direction: 选择是输入生成网络的图 是 低光照图（A） 还是 预增强的图（A_enhance）

## 训练相关参数
* fineSize： 输入网络训练的 patch size
* batchSize：batch尺寸 
* lr： 学习率
* milestones： 学习率调整的epoch，例如 100,150 代表在100和150epoch时分别x0.1

## 损失函数相关
* vgg： vgg损失的权重
* K_COSINE： 余弦相似度的损失权重
* K_MSE： MSE损失的权重
* K_GL： 亮度一致性损失权重
* K_GSPA： 梯度一致性损失权重
* K_TV： 变分损失权重
* K_GA: 全局判别器对抗损失权重
* K_GP: 局部判别器对抗损失权重

## GAN相关的参数配置
* which_model_netD：判别器的网络结构，目前支持 no_norm_4  unet_dis  patch_dis 
* wgan: 是否使用 wgan
* no_lsgan: 不使用lsgan
* use_ragan: 使用相对真假损失
* hybrid_loss: 为true，则判别器也使用 相对损失的形式
* mix_argue： 是否使用mix-argue
* use_rotate_aug：判别器的输入是否为两张图，true 输出为ref和fake的cat
* full_batch_mixup: 是否全batch都为mix样本 （若使用mix只能设置为True，不支持false）
* slow_mixup：是否缓慢增加mix_batch的概率 （若使用mix只能设置为True，不支持false）
* patchD_3： 大于0 表示使用局部判别器
* patchSize： 局部判别器的输出 patch 尺寸可设置为 64 或 32
* patch_vgg： 是否使用 patch vgg
* n_layers_D: 判别器的网络层数，只针对no_norm_4判别器有效
* n_layers_patchD： 局部判别器层数

> 不同类型的 GAN 参数配置 \
wgan: use_wgan=true  no_lsgan=false(对抗网络输出不加sigmoid) \
lsgan：use_wgan=false no_lsgan=false  \
ragan: use_wgan=false no_lsgan=false  use_ragan=true   hybrid_loss=False(对抗器也使用ra的形式) True(不使用ra的形式)\
cgan：use_rotate_aug=True(对抗网络输入为6通道，否则为3通道)  aligen_AB=False(设置AB是否来自成对的图象)\
unet: which_model_netD='unet_dis'/'no_norm_4'(对抗网络为unet或常规的patch网络) \
mixup: mix_argue=True full_batch_mixup=True slow_mixup=True(和unet判别器一起使用的正则手段，注意mix的其他参数)

# 数据制作
1. 将原始数据集四倍下采样。 tools/batch_resize.py 
2. 将下采样后的数据，打包成.h5文件。  tools/make_hd5f.py
3. 将lableme软件标注生成的json注释打包。 tools/load_json_file.py

完成上述步骤后将生成的 .h5数据和.json数据放在同一目录下，即完成训练数据的制作。测试数据按照下述格式存放即可。
>train_data \
>-------train.h5 \
>-------box_annos.json \
>test_data \
>-------testA \
>--------------xxx_-2ev_xx1.png \
>--------------xxx_-2ev_xx2.png \
>-------testB \
>--------------xxx_0ev_xx1.png \
>--------------xxx_0ev_xx2.png

# 训练
训练的时候，先用一个终端打开 visodom： 
> python -m visdom.server -port=48135 --hostname=127.0.0.1 

然后在浏览器中打开可视化界面, 然后在另一个终端运行训练命令 
> sh run.sh \
> sh run_ohem.sh  难样本训练

训练脚本中参数含义在脚本中注释，其他配置参数有.yaml修改

# 测试
运行脚本即可，具体参数含义在 脚本文件中注释