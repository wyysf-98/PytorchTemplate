import os
import os.path as osp
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

    def __init__(self, config):
        """Init BaseTrainer Class."""
        self.config = config
        self.mode = config.mode
        self.save_dir = config.save_dir
        self.nets = dict()
        self.losses = dict()
        self.extra = dict()
        self.optimizers = dict()
        self.schedulers = dict()
        self.clock = TrainClock()

        # init distribute training or single gpu training
        self.init_device()

        # init project, create local txt and tensorboard logger
        self.init_project(config)

        if self.mode is 'train' or self.mode is 'eval':
            # save current codes
            shutil.copytree(osp.dirname(osp.dirname(os.path.abspath(__file__))), self.save_dir / 'codes', \
                ignore=shutil.ignore_patterns('data', 'outputs', '*.txt'))
            # get dataloader
            self.prepare_dataloader(config['dataset'])
            
        # build network
        self.build_model(config['model'])

        if self.mode is 'train':
            # set loss function
            self.set_loss_function(config['loss'])
            # configure optimizers
            self.configure_optimizers(config['optimizer'])

    def master_process(func):
        """ decorator for master process """
        def wrapper(self, *args, **kwargs):
            if self.is_master:
                return func(self, *args, **kwargs)
        return wrapper

    @abstractmethod
    def prepare_dataloader(self, data_config):
        """prepare dataloader for training"""
        raise NotImplementedError

    @abstractmethod
    def build_model(self, model_config):
        """build networks for training"""
        raise NotImplementedError

    @abstractmethod
    def set_loss_function(self, loss_config):
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
    def init_project(self, config):
        """Init project. Create log folder and txt/tensorboard logger."""
        os.makedirs(self.save_dir / 'log', exist_ok=True)
        # create txt logger
        self.logger = WorklogLogger(self.save_dir / 'log' / 'log.txt')
        self.logger.put_line(f'save to ======> {self.save_dir}')
        self.record_str(config)

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
            self.tb.add_scalar('{}_loss/{}'.format(prefix, k), v.item(), self.clock.step)
        if not mute:
            self.logger.put_line(
                '[Epoch/Step : {}/{}]: {}'.format(self.clock.epoch, self.clock.step, record_str))

    @master_process
    def record_scalar(self, dict_recorded, prefix=None, mute=True):
        """record scalar to tensorboard and comet_ml if use comel is True"""
        str_recorded = ''
        for k, v in dict_recorded.items():
            str_recorded += '{}: {:.8f} '.format(k, v.item())
            self.tb.add_scalar(k if prefix is None else '{}/{}'.format(prefix, k), v.item(), self.clock.step)
        if not mute:
            self.logger.put_line(
                '[Epoch/Step : {}/{}]: {}'.format(self.clock.epoch, self.clock.step, str_recorded))

    @master_process
    def record_str(self, str_recorded):
        """record string in master process"""
        print(str_recorded)
        self.logger.put_line(
            '[Epoch/Step : {}/{}]: {}'.format(self.clock.epoch, self.clock.step, str_recorded))
    
    @master_process
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(
                self.save_dir, "epoch_{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.save_dir, "{}.pth".format(name))
        
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
        torch.save(save_dict, save_path)

    def load_ckpt(self, name=None, restore_clock=True, restore_optimizer=True):
        """load checkpoint from saved checkpoint"""
        load_path = name if str(name).endswith('.pth') else '{}.pth'.format(str(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        print("Loading checkpoint from {} ...".format(load_path))

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

    def get_optimizer(self, optimizer_config, parameters):
        """set optimizer used in training"""
        eps = 1e-8
        if optimizer_config['type'] == 'sgd':
            optimizer = optim.SGD(
                parameters, lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])
        elif optimizer_config['type'] == 'adam':
            optimizer = optim.Adam(
                parameters, lr=optimizer_config['lr'], eps=eps, weight_decay=optimizer_config['weight_decay'])
        else:
            raise NotImplementedError('Optimizer type {} not implemented yet !!!'.format(
                optimizer_config['type']))

        return optimizer

    def get_scheduler(self, scheduler_config, optimizer):
        """set lr scheduler used in training"""
        eps = 1e-8
        if scheduler_config['type'] == 'steplr':
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=scheduler_config['decay_step'], gamma=scheduler_config['decay_gamma'])
        elif scheduler_config['type'] == 'explr':
            scheduler = lr_scheduler.ExponentialLR(
                optimizer, scheduler_config['lr_decay'])
        elif scheduler_config['type'] == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_config['num_epochs'], eta_min=eps)
        elif scheduler_config['type'] == 'poly':
            scheduler = lr_scheduler.LambdaLR(
                optimizer, lambda epoch: (1-epoch/scheduler_config['num_epochs'])**scheduler_config['poly_exp'])
        else:
            raise NotImplementedError('Scheduler type {} not implemented yet !!!'.format(
                scheduler_config['type']))
        return scheduler

    def configure_optimizers(self, optimizers_config):
        """configure optimizers used in training"""
        parameters = []
        for key in self.nets.keys():
            parameters += list(self.nets[key].parameters())

        optimizer = self.get_optimizer(optimizers_config, parameters)
        scheduler = self.get_scheduler(optimizers_config['scheduler'], optimizer)

        self.optimizers['base'] = optimizer
        self.schedulers['base'] = scheduler

    def update_learning_rate(self):
        """record and update learning rate"""
        def get_learning_rate(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        for key in self.optimizers.keys():
            if self.is_master:
                current_lr = get_learning_rate(self.optimizers[key])
                self.logger.put_line('[Epoch/Step : {}/{}]: <optimizer {}> learning rate is: {}'.format(
                    self.clock.epoch, self.clock.step, key, current_lr))
                self.tb.add_scalar('learning_rate/{}_lr'.format(key), current_lr, self.clock.epoch)

            self.schedulers[key].step()

    def update_network(self):
        """update network by back propagation"""
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad()

        total_loss = sum(self.losses.values())
        total_loss.backward()

        for key in self.optimizers.keys():
            self.optimizers[key].step()

    def train_func(self, data):
        """training function"""
        self.train_mode()

        self.train_step(data)
        self.update_network()

        if self.clock.step % self.config['runtime']['log_iter']==0:
            self.record_losses('train')

    def val_func(self, data):
        """validation function"""
        self.eval_mode()

        with torch.no_grad():
            self.val_step(data)

        if self.clock.step % self.config['runtime']['log_iter']==0:
            self.record_losses('valid')