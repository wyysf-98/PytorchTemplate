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

    trainer.val_loader = inf_loop(trainer.val_loader)
    for e in range(clock.epoch, config['runtime']['num_epochs']):
        # if use DDP mode, set train sampler every epoch
        if trainer.dist:
            trainer.train_sampler.set_epoch(e)
        # save init state
        if e == 1:
            trainer.save_ckpt()
            if config['trainer']['vis_epoch']:
                trainer.visualize_batch()
                
        # begin train iteration
        train_pbar = tqdm(trainer.train_loader)
        for b, data in enumerate(train_pbar):
            # train step
            trainer.train_func(data)

            # validation step
            if clock.step % config['trainer']['val_every_n_step'] == 0:
                data = next(trainer.val_loader)
                trainer.val_func(data)
                if config['trainer']['vis_step'] and \
                    clock.step % (config['trainer']['vis_every_n_val'] * \
                    config['trainer']['val_every_n_step']) == 0:
                    trainer.visualize_batch()

            # set pbar
            train_pbar.set_description("Train EPOCH[{}][{}]".format(e, b))
            log_dict = {**trainer.losses, **trainer.extra}
            train_pbar.set_postfix(OrderedDict({k: '%.4f'%v.item()
                                        for k, v in log_dict.items()}))
            # clock tick
            clock.tick()

        # update learning rate
        trainer.update_learning_rate()
        # clock tock
        clock.tock()

        # save checkpoint
        if clock.epoch % config['trainer']['save_every_n_epoch'] == 0 or clock.epoch == 1:
            trainer.save_ckpt()
        trainer.save_ckpt('latest')  

        # vis batch
        if config['trainer']['vis_epoch'] and clock.epoch % config['trainer']['vis_every_n_epoch'] == 0:
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
