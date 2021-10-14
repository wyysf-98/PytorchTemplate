import torch.nn as nn
from .losses import VGGPerceptual_L1Loss, SSIM_L1Loss, L1Loss, SSIMLoss

def get_model(model_config):    
    if model_config['type'] == "sss":
        pass
    else:
        raise ValueError('model type: {} not valid'.format(model_config['type']))

def get_loss(loss_config):    
    if loss_config['type'] == "VGGPerceptual_L1":
        return VGGPerceptual_L1Loss()
    elif loss_config['type'] == "SSIM_L1":
        return SSIM_L1Loss()
    elif loss_config['type'] == "L1":
        return L1Loss()
    elif loss_config['type'] == "SSIM":
        return SSIMLoss()
    else:
        raise ValueError('loss type: {} not valid'.format(loss_config['type']))
