
def create_model(opt):
    ''' 根据模型名称创建模型，目前支持三种类型的方法
    restore_model：常规的损失函数，端到端训练
    restore_gan_model： 基于gan方法的模型，整合了所有尝试过的gan方法
    fusion_model： 基于多阶段的 融合模型
    '''
    model = None
    if opt.model == 'restore_model':
        from .restore_model import RestoreModel
        model = RestoreModel()
    elif opt.model == 'restore_gan_model':
        from .restore_gan_model import RestoreGANModel
        model = RestoreGANModel()
    elif opt.model == 'fusion_model':
            from .restore_fusion_model import FusionModel
            model = FusionModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
