import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.mydist import is_main_rank
from tqdm import tqdm

''' 训练文件
    通过调用 /models 文件夹下的 restore_model.py完成训练
'''

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

mainRank = is_main_rank()

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    # data_loader.shuffle_data(epoch)
    epoch_start_time = time.time()

    if mainRank:
        t_bar = tqdm(dataset)
        t_bar.set_description('Epoch: [{}]'.format(epoch))
    else:
        t_bar = dataset

    for data in t_bar:
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters(epoch)

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
        