import imp
import torch.nn as nn

def get_model(model_cfg):    
    model_module = 'models.' + model_cfg.type
    model_path = 'models/' + model_cfg.type + '.py'
    return imp.load_source(model_module, model_path).Model(model_cfg)

def get_loss(loss_config):    
    pass