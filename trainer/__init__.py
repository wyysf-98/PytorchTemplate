import imp

def get_trainer(cfg):
    trainer_module = 'trainer.' + cfg.model.type
    trainer_path = 'trainer/' + cfg.model.type + '.py'
    return imp.load_source(trainer_module, trainer_path).Trainer(cfg)