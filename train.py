import os
import os.path as osp
import torch
import argparse
from tqdm import tqdm
from collections import OrderedDict
from trainer import get_trainer
from utils.tools import inf_loop
from utils.config_parser import ConfigParser

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train(config):
    # create network and trainer
    trainer = get_trainer(config)
    
    # load from checkpoint if provided
    if config.resume:
        trainer.load_ckpt(config.resume)

    # create training clock
    clock = trainer.clock

    if config['val_mode'] == 'iter':
        trainer.datas['val'] = inf_loop(trainer.datas['val'])
    
    for e in range(clock.epoch, config['num_epochs']):
        # save init state
        if e == 1:
            trainer.save_ckpt()
            try:
                trainer.visualize_batch()
            except Exception as ex:
                print(f'WARNING: {ex}, do not support vis first epoch.')
                
        # begin train iteration
        train_pbar = tqdm(trainer.datas['train'])
        for b, data in enumerate(train_pbar):
            # train step
            trainer.train_func(data)
            # set pbar
            train_pbar.set_description("Train EPOCH[{}][{}]".format(e, b))
            log_dict = {**trainer.losses, **trainer.extra}
            train_pbar.set_postfix(OrderedDict({k: '%.4f'%v.item() for k, v in log_dict.items()}))
            # clock tick
            clock.tick()
            # validation step
            if config['val_mode'] == 'iter' and clock.step % config['val_every_n_iter'] == 0:
                data = next(trainer.datas['val'])
                trainer.val_func(data)
                train_pbar.set_description("Val EPOCH[{}][{}]".format(e, b))
                log_dict = {**trainer.losses, **trainer.extra}
                train_pbar.set_postfix(OrderedDict({k: '%.4f'%v.item() for k, v in log_dict.items()}))
                if clock.step % (config['vis_every_n_val'] * config['val_every_n_iter']) == 0:
                    trainer.visualize_batch()
      
        # update learning rate
        trainer.update_learning_rate()
        # clock tock
        clock.tock()

        # save checkpoint
        if clock.epoch % config['save_every_n_epoch'] == 0:
            trainer.save_ckpt()
        trainer.save_ckpt('latest')  

        # begin val iteration
        if config['val_mode'] == 'epoch' and clock.epoch % config['val_every_n_epoch'] == 0:
            val_pbar = tqdm(trainer.datas['val'])
            for b, data in enumerate(val_pbar):
                # val step
                trainer.val_func(data)
                # set pbar
                val_pbar.set_description("Val EPOCH[{}][{}]".format(e, b))
                log_dict = {**trainer.losses, **trainer.extra}
                val_pbar.set_postfix(OrderedDict({k: '%.4f'%v.item() for k, v in log_dict.items()}))
                # vis batch
                if clock.epoch % (config['vis_every_n_val'] * config['val_every_n_epoch']) == 0:
                    trainer.visualize_batch()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='training pipeline defination')
    args.add_argument('-m', '--mode', default='train', type=str, help='current mode. ')
    args.add_argument('-c', '--config', required=True, type=str, help='config file path.')
    args.add_argument('-r', '--resume', default=None, type=str, help='file path to retore the checkpoint. (default: None)')
    args.add_argument('-n', '--name', default=None, type=str, help='job name. If None, use current time stamp. (default: None)')
    args.add_argument('-s', '--seed', default=None, help='random seed used. (default: None)')
    args.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    args.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")

    config = ConfigParser(args)

    train(config)
