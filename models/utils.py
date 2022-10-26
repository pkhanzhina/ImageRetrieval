import torch


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        module.eval()
        module.train = lambda _: None