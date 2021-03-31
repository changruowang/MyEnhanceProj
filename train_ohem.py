#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 难样本训练文件，通过调用 /models 文件夹下的 restore_model.py完成训练，训练逻辑稍和 
                 train不同，所以单独写了一个文件主要是多了一个前向计算损失的过程。
@Date          : 2021/03/29 14:25:47
@Author        : changruowang
@version       : 1.0
'''
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.image_pool import HardExampalePool
from util.mydist import is_main_rank
from tqdm import tqdm
import torch

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

hard_image_pool = HardExampalePool(int(dataset_size * 0.5))
hard_dataset = torch.utils.data.DataLoader(
                        hard_image_pool,### 难样本的比例 
                        batch_size=opt.batchSize // 2,
                        shuffle=not opt.serial_batches,
                        num_workers=int(opt.nThreads))

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
mainRank = is_main_rank()


def set_data(epoch, backward=True):
    if not backward:
        data_tmp, str_ = dataset, 'forward'
    else:
        data_tmp, str_ = hard_dataset, 'backward'

    if mainRank:
        t_bar = tqdm(data_tmp)
        t_bar.set_description('Epoch: [{}-{}] forward'.format(epoch, str_))
    else:
        t_bar = data_tmp

    return t_bar

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    # data_loader.shuffle_data(epoch)
    epoch_start_time = time.time()
    
    t_bar = set_data(epoch, backward=False)
### 前向一遍数据集
    with torch.no_grad():
        for data in t_bar:
            model.set_input(data)
            model.optimize_parameters(epoch, is_backward=False, image_pool=hard_image_pool)

    t_bar = set_data(epoch, backward=True)

###选择难样本后向
    for data in t_bar:
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)

        model.optimize_parameters(epoch, is_backward=True)

        if mainRank:   ### 主进程刷新显示
            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors(epoch)
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

    model.update_learning_rate()

    if mainRank:
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
            model.save_networks(epoch)
        
