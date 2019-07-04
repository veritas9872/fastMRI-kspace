import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tensorboardX import SummaryWriter

from time import time
from pathlib import Path

from train.subsample import MaskFunc
from utils.train_utils import CheckpointManager
from utils.run_utils import get_logger, initialize, save_dict_as_json
from utils.modelsummary import summary
from data.mri_data import SliceData
from data.pre_processing import TrainInputSliceTransform, NewTrainInputSliceTransform, TrainInputBatchTransform

from models.k_unet_model import UnetModel  # TODO: Create method to specify model in main.py


def create_datasets(args):
    if args.batch_size > 1:
        raise NotImplementedError('Batch size must be greater than 1 for the current implementation to function.')

    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    divisor = 2 ** args.num_pool_layers  # UNET architecture requires that all inputs be dividable by some power of 2.
    # The current implementation only works if the batch size is 1.

    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=TrainInputSliceTransform(train_mask_func, args.challenge, use_seed=False, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=TrainInputSliceTransform(val_mask_func, args.challenge, use_seed=True, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    return train_dataset, val_dataset


def create_data_loaders(args):
    train_dataset, val_dataset = create_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def create_new_data_loaders(args, device):
    if args.batch_size > 1:
        raise NotImplementedError('Batch size must be greater than 1 for the current implementation to function.')

    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    divisor = 2 ** args.num_pool_layers  # UNET architecture requires that all inputs be dividable by some power of 2.
    # The current implementation only works if the batch size is 1.

    # Generating Datasets.
    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=NewTrainInputSliceTransform(
            train_mask_func, args.challenge, device, use_seed=False, amp_fac=args.amp_fac,
            divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=NewTrainInputSliceTransform(
            val_mask_func, args.challenge, device, use_seed=True, amp_fac=args.amp_fac, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False  # Since tensors are already on GPU.
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False  # Since tensors are already on GPU.
    )
    return train_loader, val_loader


def create_batch_data_loaders(args, device):
    if args.batch_size > 1:
        raise NotImplementedError('Batch size must be greater than 1 for the current implementation to function.')

    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    divisor = 2 ** args.num_pool_layers  # UNET architecture requires that all inputs be dividable by some power of 2.
    # The current implementation only works if the batch size is 1.

    # Generating Datasets.
    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=TrainInputBatchTransform(
            train_mask_func, args.challenge, device, use_seed=False, amp_fac=args.amp_fac,
            divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=TrainInputBatchTransform(
            val_mask_func, args.challenge, device, use_seed=True, amp_fac=args.amp_fac, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False  # Since tensors are already on GPU.
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False  # Since tensors are already on GPU.
    )
    return train_loader, val_loader


class Trainer(object):
    def __init__(self, args, model=None, optimizer=None, loss_func=None, metrics=None, scheduler=None):
        # TODO: I found that the loss disappears due to numerical underflow when trained like this.
        #  I need to implement multiplication to make training more stable.
        #  I have verified that training is possible.

        self.args = args
        # assert model is not None

        if self.args.batch_size > 1:
            raise NotImplementedError('Only batch size of 1 for now.')

        ckpt_path = Path(self.args.ckpt_dir)
        ckpt_path.mkdir(exist_ok=True)

        run_number, run_name = initialize(ckpt_path)

        ckpt_path = ckpt_path / run_name
        ckpt_path.mkdir(exist_ok=True)

        log_path = Path(self.args.log_dir)
        log_path.mkdir(exist_ok=True)
        log_path = log_path / run_name
        log_path.mkdir(exist_ok=True)

        save_dict_as_json(vars(self.args), log_dir=log_path, save_name=run_name)

        self.logger = get_logger(name=__name__, save_file=log_path / run_name)

        # Assignment inside running code appears to work.
        if (self.args.gpu is not None) and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.args.gpu}')
            self.logger.info(f'Using GPU {self.args.gpu} for {run_name}')
        else:
            self.device = torch.device('cpu')
            self.logger.info(f'Using CPU for {run_name}')

        # Create Datasets. Use one slice at a time for now.
        self.train_loader, self.val_loader = create_batch_data_loaders(self.args, self.device)

        if model is None:
            # Define model.
            data_chans = 2 if self.args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
            # data_chans indicates the number of channels in the data.
            self.model = UnetModel(in_chans=data_chans, out_chans=data_chans, chans=self.args.chans,
                                   num_pool_layers=self.args.num_pool_layers).to(self.device)  # TODO: Move to main.py
        else:
            self.model = model.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.init_lr)  # Maybe move to main.py
        else:
            self.optimizer = optimizer

        if loss_func is None:
            self.loss_func = nn.L1Loss(reduction='mean').to(self.device)
        else:
            self.loss_func = loss_func.to(self.device)

        self.metrics = metrics

        self.checkpointer = CheckpointManager(model=self.model, optimizer=self.optimizer, mode='min',
                                              save_best_only=self.args.save_best_only,
                                              ckpt_dir=ckpt_path, max_to_keep=self.args.max_to_keep)

        if hasattr(self.args, 'previous_model') and self.args.previous_model:
            self.checkpointer.load(load_dir=self.checkpointer, load_optimizer=False)

        if scheduler is None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(  # TODO: This should also be moved to main.py
                self.optimizer, mode='min', factor=0.1, patience=5, verbose=True, cooldown=0, min_lr=1E-7)
        else:
            self.scheduler = scheduler

        self.writer = SummaryWriter(log_dir=str(log_path))

    def train_model(self):
        torch.multiprocessing.set_start_method("spawn")
        summary(model=self.model, input1_size=(30, 640, 384), input2_size=(15, 640, 378),
                batch_size=self.args.batch_size, device=self.device, display_func=self.logger.info)
        self.logger.info('Beginning Training loop')
        for epoch in range(1, self.args.num_epochs + 1):  # 1 based indexing
            # Training
            tic = time()
            train_epoch_loss, train_epoch_metrics = self._train_epoch(epoch)
            toc = int(time() - tic)

            self._print_state(epoch, train_epoch_loss, toc, train_epoch_metrics, training=True)

            # Evaluating
            tic = time()
            val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch)
            toc = int(time() - tic)

            self._print_state(epoch, val_epoch_loss, toc, val_epoch_metrics, training=False)

            for idx, group in enumerate(self.optimizer.param_groups, start=1):
                self.writer.add_scalar(f'learning_rate_{idx}', group['lr'], epoch)

            self.checkpointer.save(metric=val_epoch_loss, verbose=True)

            if self.scheduler is not None:
                self.scheduler.step(metrics=val_epoch_loss)

        return self.model

    def _train_epoch(self, epoch):
        self.model.train()
        torch.autograd.set_grad_enabled(True)

        # initialize epoch loss
        epoch_loss_lst = list()  # Appending values to list due to numerical underflow.
        # initialize multiple epoch metrics
        epoch_metrics_lst = [list() for _ in self.metrics] if self.metrics else None

        # labels are fully sampled coil-wise images, not rss or esc.
        for idx, (inputs, targets) in enumerate(self.train_loader, start=1):

            # Data should be in CUDA already, not be sent to GPU here. Otherwise, ifft2d for labels will be slow!!!
            # inputs = inputs.to(device) * 10000  # TODO: Fix this later!! Very ugly hack...
            # targets = targets.to(device) * 10000  # TODO: Fix this later.
            # # NOTE: This x10000 is here because the values themselves need to be amplified.
            step_loss, recons = self._train_step(inputs, targets)

            # Gradients are not calculated so as to boost speed and remove weird errors.
            with torch.no_grad():  # Update epoch loss and metrics
                epoch_loss_lst.append(step_loss.item())  # Perhaps not elegant, but underflow makes this necessary.
                if self.metrics:
                    step_metrics = [metric(recons, targets) for metric in self.metrics]
                    for step_metric, epoch_metric_lst in zip(step_metrics, epoch_metrics_lst):
                        epoch_metric_lst.append(step_metric.item())

                if self.args.verbose:
                    print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item():.4e}')
                    if self.metrics:
                        for step_metric in step_metrics:
                            print(f'Training metric Epoch {epoch:03d} Step {idx:03d}: {step_metric.item():.4e}')

        epoch_loss = float(np.nanmean(epoch_loss_lst))  # Remove nan values just in case.
        epoch_metrics = [float(np.nanmean(epoch_metric_lst)) for epoch_metric_lst in
                         epoch_metrics_lst] if self.metrics else None

        num_nans = np.isnan(epoch_loss_lst).sum()
        if num_nans > 0:
            print(f'Epoch {epoch} training: {num_nans} NaN values present in {len(self.train_loader.dataset)} slices')

        return epoch_loss, epoch_metrics

    def _train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        recons = self.model(inputs, targets)
        step_loss = self.loss_func(recons, targets)
        step_loss.backward()
        self.optimizer.step()
        return step_loss, recons

    def _val_epoch(self, epoch):
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss_lst = list()
        epoch_metrics_lst = [list() for _ in self.metrics] if self.metrics else None

        for step, (inputs, targets) in enumerate(self.val_loader, start=1):
            # inputs = inputs.to(device) * 10000  # TODO: Fix this later!! VERY UGLY HACK!!
            # targets = targets.to(device) * 10000  # TODO: Remove later!
            step_loss, recons = self._val_step(inputs, targets)

            with torch.no_grad():  # Probably not actually necessary...
                epoch_loss_lst.append(step_loss.item())
                if self.metrics:
                    step_metrics = [metric(recons, targets) for metric in self.metrics]
                    for step_metric, epoch_metric_lst in zip(step_metrics, epoch_metrics_lst):
                        epoch_metric_lst.append(step_metric.item())

                if self.args.verbose:
                    print(f'Epoch {epoch:03d} Step {step:03d} Validation loss: {step_loss.item():.4e}')
                    if self.metrics:
                        for idx, step_metric in enumerate(step_metrics):
                            print(
                                f'Epoch {epoch:03d} Step {step:03d}: Validation metric {idx}: {step_metric.item():.4e}')

                if self.args.max_imgs:
                    interval = len(self.val_loader.dataset) // self.args.max_imgs
                    if step % interval == 0:  # Note that all images are scaled independently of all other images.
                        assert isinstance(self.writer, SummaryWriter)
                        recons_grid, targets_grid, deltas_grid = self._make_grid_triplet(recons, targets)
                        self.writer.add_image(f'Recons', recons_grid, epoch, dataformats='HW')
                        self.writer.add_image(f'Targets', targets_grid, epoch, dataformats='HW')
                        self.writer.add_image(f'Deltas', deltas_grid, epoch, dataformats='HW')

        epoch_loss = float(np.nanmean(epoch_loss_lst))  # Remove nan values just in case.
        epoch_metrics = [float(np.nanmean(epoch_metric_lst)) for epoch_metric_lst in epoch_metrics_lst] if self.metrics else None

        num_nans = np.isnan(epoch_loss_lst).sum()
        if num_nans > 0:
            print(f'Epoch {epoch} validation: {num_nans} NaN values present in {len(self.val_loader.dataset)} slices')

        return epoch_loss, epoch_metrics

    def _val_step(self, inputs, targets):
        recons = self.model(inputs, targets)
        step_loss = self.loss_func(recons, targets)
        return step_loss, recons

    def _make_grid_triplet(self, recons, targets):
        assert recons.shape == targets.shape
        if recons.size(0) > 1:
            raise NotImplementedError('Mini-batch size greater than 1 has not been implemented yet.')

        # Assumes batch size of 1.
        recons = recons.detach().cpu().squeeze(dim=0)
        targets = targets.detach().cpu().squeeze(dim=0)

        large = torch.max(targets)
        small = torch.min(targets)
        diff = large - small

        # Scaling to 0~1 range.
        recons = (recons.clamp(min=small, max=large) - small) / diff
        targets = (targets - small) / diff

        if recons.size(0) == 15:
            recons = torch.cat(torch.chunk(recons.view(size=(-1, recons.size(-1))), chunks=5, dim=0), dim=1)
            targets = torch.cat(torch.chunk(targets.view(size=(-1, targets.size(-1))), chunks=5, dim=0), dim=1)
        elif recons.size(0) == 1:
            recons = recons.squeeze()
            targets = targets.squeeze()
        else:
            raise ValueError('Invalid dimensions!')

        deltas = targets - recons

        return recons, targets, deltas

    def _print_state(self, epoch, epoch_loss, toc, metric, training=True):
        which_state = 'Training' if training else 'Validation'
        which_loss = 'train_epoch_loss' if training else 'val_epoch_loss'
        which_metric = 'train_epoch_metric' if training else 'val_epoch_metric'

        self.logger.info(
            f'Epoch {epoch:03d} {which_state}. loss: {epoch_loss:.4e}, Time: {toc // 60}min {toc % 60}sec')
        self.writer.add_scalar(which_loss, scalar_value=epoch_loss, global_step=epoch)

        if isinstance(metric, list):  # The metrics being returned are either 'None' or a list of values.
            for idx, epoch_metric in enumerate(metric, start=1):
                self.logger.info(f'Epoch {epoch:03d} {which_state}. Metric {idx}: {epoch_metric}')
                self.writer.add_scalar(f'{which_metric}_{idx}', scalar_value=epoch_metric, global_step=epoch)
