import torch.distributed as dist

### 多卡训练
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_model_param(model):
    if is_dist_avail_and_initialized():
        if get_rank() == 0:
            return model.module.state_dict()
    else:
        return model.state_dict()

def is_main_rank():
    if is_dist_avail_and_initialized():
        return get_rank() == 0
    return True