import random
import numpy as np
import torch
import heapq as hp

class ImagePool():
    ''' 用于将训练过程生成的fake图象缓存到一个列表，每次训练判别器，从缓冲池中随机选一个训练判别器。
        本项目的gan代码中未使用
    '''
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image.cpu())
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone().to(image.device)
                    self.images[random_id] = image.cpu()
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

### 根据Loss 大小记录难样本  因为每个 epoch 训练样本crop的可能不一样 所以要直接记录crop后的难样本

class  ImageNode():
    ''' 保存 难样本 的基本节点
    Args:
        img_dict： 一个训练样本
        loss： 训练样本对应的损失 标量
    '''
    def __init__(self, img_dict, loss):
        self.loss = loss
        self.img_dict = img_dict

    def __lt__(self, other):
        ''' 重载了 大小判断，使得自定义的类可以用 python 的优先队列
        '''
        if self.loss < other.loss:
            return True
        else:
            return False

    def get_loss(self):
        return self.loss

    def __call__(self):
        return self.img_dict

class HardExampalePool():
    ''' 难样本挖掘中使用，数据缓冲区，用于从前向传播的数据流中筛选 loss 前百分之N大的样本缓存，
        在后向传播时将该类 作为 daloader　基础　使用缓存的难样本训练
    Args: 
        pool_size : 样本池的容量，代表 topK 难样本
    '''
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.hard_images = []

    def add_example(self, batch_data, loss_vec):
        ''' 前向传播中， 将样本和对应的损失 输出，然后该函数将其中损失将其中较大的样本缓存
        Agrs：
            batch_data: 为 一个batch的训练样本，字典，格式和 HD5F 读取的样本一样
            loss_vec： 损失向量，对应每个样本的损失 loss_vec.shape[0] == len(batch_data[key]) 
        '''
        losses = loss_vec.cpu()
        b = losses.shape[0]
        for i in range(b):
            loss = losses[i].item()

            data_dict = {}
            for k,v in batch_data.items():
                data_dict[k] = v[i,:,:,:]
            ### 优先队列 筛选 topK
            if self.num_imgs < self.pool_size:
                hp.heappush(self.hard_images, ImageNode(data_dict, loss))
                self.num_imgs+=1  
            elif loss >= self.hard_images[0].get_loss():
                hp.heapreplace(self.hard_images, ImageNode(data_dict, loss)) 
      
    def __getitem__(self, index):
        ''' 反向传播， 输出一个样本，和 pytorch dataloader配合使用的
        '''
        return self.hard_images[index % len(self.hard_images)]()
                   
               
    def __len__(self):
        return self.pool_size
