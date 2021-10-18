import os
import os.path as osp
from pathlib import Path
import shutil
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from accelerate import Accelerator
from torch.nn.parallel.distributed import DistributedDataParallel
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.tools import TrainClock, WorklogLogger

class BaseTrainer(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """

    def __init__(self, cfg):
        """Init BaseTrainer Class."""
        self.cfg = cfg
        self.mode = cfg.mode
        self.save_dir = cfg.save_dir
        self.datas = dict()
        self.nets = dict()
        self.losses = dict()
        self.extra = dict()
        self.optimizers = dict()
        self.schedulers = dict()
        self.clock = TrainClock()

        # init distribute training or single gpu training
        self.init_device()

        # init project, create local txt and tensorboard logger
        self.init_project(cfg)

        if self.mode is 'train' or self.mode is 'eval':
            if self.is_master:
                # save current codes
                shutil.copytree(Path(__file__).parent.parent, self.save_dir / 'codes', \
                    ignore=shutil.ignore_patterns('data', 'outputs', '*.txt', '.*'))
            # get dataloader
            self.prepare_dataloader(cfg['dataset'])

        # build network
        self.build_model(cfg['model'])

        if self.mode is 'train':
            # set loss function
            self.set_loss_function(cfg['loss'])
            # cfgure optimizers
            self.cfgure_optimizers(cfg['optimizer'])

        # prepare for accelerator
        for m in [self.datas, self.nets, self.optimizers]:
            for key in m.keys():
                m[key] = self.accelerator.prepare(m[key])

    def master_process(func):
        """ decorator for master process """
        def wrapper(self, *args, **kwargs):
            if self.is_master:
                return func(self, *args, **kwargs)
        return wrapper

    @abstractmethod
    def prepare_dataloader(self, data_cfg):
        """prepare dataloader for training"""
        raise NotImplementedError

    @abstractmethod
    def build_model(self, model_cfg):
        """build networks for training"""
        raise NotImplementedError

    @abstractmethod
    def set_loss_function(self, loss_cfg):
        """set loss function used in training"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, data):
        """forward logic in network"""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, data):
        """one step of training"""
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data):
        """one step of validation"""
        raise NotImplementedError

    @abstractmethod
    def visualize_batch(self):
        """visualize results"""
        raise NotImplementedError

    def init_device(self):
        """init devices. Use accelerator to support single/multi GPU, TPU etc"""
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.is_master = self.accelerator.is_main_process
        self.local_rank = self.accelerator.process_index

    @master_process
    def init_project(self, cfg):
        """Init project. Create log folder and txt/tensorboard logger."""
        os.makedirs(self.save_dir / 'log', exist_ok=True)
        # create txt logger
        self.logger = WorklogLogger(self.save_dir / 'log' / 'log.txt')
        self.logger.put_line(f'save to ======> {self.save_dir}')
        self.record_str(cfg)

        # set tensorboard writer
        self.tb = SummaryWriter(self.save_dir / 'log' / 'train.events')

    @master_process
    def record_losses(self, prefix='train', mute=False):
        """record loss to tensorboard. print if mute is not True"""
        record_str = ''
        dict_recorded = {**self.losses, **self.extra}
        dict_recorded['total'] = sum(self.losses.values())
        for k, v in dict_recorded.items():
            record_str += '{}: {:.8f} '.format(k, v.item())
            self.tb.add_scalar(f'{prefix}_loss/{k}', v.item(), self.clock.step)
        if not mute:
            self.logger.put_line(
                f'[Epoch/Step : {self.clock.epoch}/{self.clock.step}]: {record_str}')

    @master_process
    def record_scalar(self, dict_recorded, prefix=None, mute=True):
        """record scalar to tensorboard and comet_ml if use comel is True"""
        str_recorded = ''
        for k, v in dict_recorded.items():
            str_recorded += '{}: {:.8f} '.format(k, v.item())
            self.tb.add_scalar(k if prefix is None else f'{prefix}/{k}', v.item(), self.clock.step)
        if not mute:
            self.logger.put_line(
                f'[Epoch/Step : {self.clock.epoch}/{self.clock.step}]: {str_recorded}')

    @master_process
    def record_str(self, str_recorded):
        """record string in master process"""
        print(str_recorded)
        self.logger.put_line(
            f'[Epoch/Step : {self.clock.epoch}/{self.clock.step}]: {str_recorded}')
    
    @master_process
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(
                self.save_dir, f"epoch_{self.clock.epoch}.pth")
            print(f"Saving checkpoint epoch {self.clock.epoch}...")
        else:
            save_path = os.path.join(self.save_dir, f"{name}.pth")
        
        save_dict = dict()
        save_dict['clock'] = self.clock.make_checkpoint()
        for key in self.nets.keys():
            if isinstance(self.nets[key], DistributedDataParallel):
                save_dict[key+'_net'] = self.nets[key].module.state_dict()
            else:
                save_dict[key+'_net'] = self.nets[key].state_dict()
        for key in self.optimizers.keys():
            save_dict[key+'_optimizer'] = self.optimizers[key].state_dict()
            save_dict[key+'_scheduler'] = self.schedulers[key].state_dict()
        self.accelerator.save(save_dict, save_path)

    def load_ckpt(self, name=None, restore_clock=True, restore_optimizer=True):
        """load checkpoint from saved checkpoint"""
        load_path = name if str(name).endswith('.pth') else f'{name}.pth'
        if not os.path.exists(load_path):
            raise ValueError(f"Checkpoint {load_path} not exists.")

        checkpoint = torch.load(load_path, map_location=self.device)
        print(f"Loading checkpoint from {load_path} ...")

        for key in self.nets.keys():
            if isinstance(self.nets[key], DistributedDataParallel):
                self.nets[key].module.load_state_dict(checkpoint[key+'_net'])
            else:
                self.nets[key].load_state_dict(checkpoint[key+'_net'])
        if restore_clock:
            self.clock.restore_checkpoint(checkpoint['clock'])
        if restore_optimizer:
            for key in self.optimizers.keys():
                if key+'_optimizer' not in checkpoint.keys():
                    self.record_str(key+'_optimizer not exist in checkpoint.')
                    continue
                self.optimizers[key].load_state_dict(checkpoint[key+'_optimizer'])
            for key in self.schedulers.keys():
                if key+'_scheduler' not in checkpoint.keys():
                    self.record_str(key+'_scheduler not exist in checkpoint.')
                    continue
                self.schedulers[key].load_state_dict(checkpoint[key+'_scheduler'])

    def get_optimizer(self, optimizer_cfg, parameters):
        """set optimizer used in training"""
        eps = 1e-8
        if optimizer_cfg['type'] == 'sgd':
            optimizer = optim.SGD(
                parameters, lr=optimizer_cfg['lr'], momentum=optimizer_cfg['momentum'], weight_decay=optimizer_cfg['weight_decay'])
        elif optimizer_cfg['type'] == 'adam':
            optimizer = optim.Adam(
                parameters, lr=optimizer_cfg['lr'], eps=eps, weight_decay=optimizer_cfg['weight_decay'])
        else:
            raise NotImplementedError(f'Optimizer type {optimizer_cfg.type} not implemented yet !!!')

        return optimizer

    def get_scheduler(self, scheduler_cfg, optimizer):
        """set lr scheduler used in training"""
        eps = 1e-8
        if scheduler_cfg['type'] == 'steplr':
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=scheduler_cfg['decay_step'], gamma=scheduler_cfg['decay_gamma'])
        elif scheduler_cfg['type'] == 'explr':
            scheduler = lr_scheduler.ExponentialLR(
                optimizer, scheduler_cfg['lr_decay'])
        elif scheduler_cfg['type'] == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_cfg['num_epochs'], eta_min=eps)
        elif scheduler_cfg['type'] == 'poly':
            scheduler = lr_scheduler.LambdaLR(
                optimizer, lambda epoch: (1-epoch/scheduler_cfg['num_epochs'])**scheduler_cfg['poly_exp'])
        else:
            raise NotImplementedError(f'Scheduler type {scheduler_cfg.type} not implemented yet !!!')
        return scheduler

    def cfgure_optimizers(self, optimizers_cfg):
        """cfgure optimizers used in training"""
        parameters = []
        for key in self.nets.keys():
            parameters += list(self.nets[key].parameters())

        optimizer = self.get_optimizer(optimizers_cfg, parameters)
        self.optimizers['base'] = optimizer

        if optimizers_cfg['scheduler'] is not None:
            scheduler = self.get_scheduler(optimizers_cfg['scheduler'], optimizer)
            self.schedulers['base'] = scheduler

    def update_learning_rate(self):
        """record and update learning rate"""
        def get_learning_rate(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        for key in self.optimizers.keys():
            if self.is_master:
                current_lr = get_learning_rate(self.optimizers[key])
                self.logger.put_line(\
                    f'[Epoch/Step : {self.clock.epoch}/{self.clock.step}]: <optimizer {key}> learning rate is: {current_lr}')
                self.tb.add_scalar(f'learning_rate/{key}_lr', current_lr, self.clock.epoch)

            self.schedulers[key].step()

    def update_network(self):
        """update network by back propagation"""
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad()

        total_loss = sum(self.losses.values())
        self.accelerator.backward(total_loss)

        for key in self.optimizers.keys():
            self.optimizers[key].step()

    def set_network(self, mode='train'):
        """set networks to train/eval mode"""
        for key in self.nets.keys():
            if mode is 'train':
                if isinstance(self.nets[key], DistributedDataParallel):
                    self.nets[key] = self.nets[key].module.train()
                else:
                    self.nets[key] = self.nets[key].train()
            elif mode is 'eval':
                if isinstance(self.nets[key], DistributedDataParallel):
                    self.nets[key] = self.nets[key].module.eval()
                else:
                    self.nets[key] = self.nets[key].eval()
            else:
                raise ValueError(f'Networks mode: {mode} not support !!!')

    def train_func(self, data):
        """training function"""
        self.set_network('train')

        self.train_step(data)
        self.update_network()

        if self.clock.step % self.cfg['log_iter']==0:
            self.record_losses('train')

    def val_func(self, data):
        """validation function"""
        self.set_network('eval')

        with torch.no_grad():
            self.val_step(data)

        if self.clock.step % self.cfg['log_iter']==0:
            self.record_losses('valid')