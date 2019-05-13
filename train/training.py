import torch
import torchvision
import torchsummary
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
from data.mri_data import SliceData
from data.slice_transforms import TrainSliceTransform

from models.k_unet_model import UnetModel

"""
Please note a bit of terminology. 
In this file, 'recons' indicates coil-wise reconstructions,
not final reconstructions for submissions.
"""


def create_datasets(args):
    if args.batch_size > 1:
        raise NotImplementedError('Batch size must be greater than 1 for the current implementation to function.')

    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    divisor = 2 ** args.num_pool_layers  # UNET architecture requires that all inputs be dividable by some power of 2.
    # The current implementation only works if the batch size is 1.

    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=TrainSliceTransform(train_mask_func, args.challenge, use_seed=False, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=TrainSliceTransform(val_mask_func, args.challenge, use_seed=True, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    return train_dataset, val_dataset


def create_data_loaders(args):
    train_dataset, val_dataset = create_datasets(args)

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


def train_step(model, optimizer, loss_func, inputs, targets):
    optimizer.zero_grad()
    recons = model(inputs, targets.shape)
    step_loss = loss_func(recons, targets)  # Pytorch uses (input, target) ordering.
    step_loss.backward()
    optimizer.step()
    return step_loss, recons


def train_epoch(model, optimizer, loss_func, data_loader, device, epoch, verbose=True, metrics=None):
    model.train()
    torch.autograd.set_grad_enabled(True)

    # initialize epoch loss
    epoch_loss_lst = list()  # Appending values to list due to numerical underflow.
    # initialize multiple epoch metrics
    epoch_metrics_lst = [list() for _ in metrics] if metrics else None

    # labels are fully sampled coil-wise images, not rss or esc.
    for idx, (inputs, targets) in enumerate(data_loader, start=1):

        inputs = inputs.to(device)
        targets = targets.to(device, non_blocking=True)
        step_loss, recons = train_step(model, optimizer, loss_func, inputs, targets)

        # Gradients are not calculated so as to boost speed and remove weird errors.
        with torch.no_grad():  # Update epoch loss and metrics
            epoch_loss_lst.append(step_loss.item())  # Perhaps not elegant, but underflow makes this necessary.
            if metrics:
                step_metrics = [metric(recons, targets) for metric in metrics]
                for step_metric, epoch_metric_lst in zip(step_metrics, epoch_metrics_lst):
                    epoch_metric_lst.append(step_metric.item())

            if verbose:
                print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item():.4e}')
                if metrics:
                    for step_metric in step_metrics:
                        print(f'Training metric Epoch {epoch:03d} Step {idx:03d}: {step_metric.item():.4e}')

    epoch_loss = np.nanmean(epoch_loss_lst)  # Remove nan values just in case.
    epoch_metrics = [np.nanmean(epoch_metric_lst) for epoch_metric_lst in epoch_metrics_lst] if metrics else None

    num_nans = np.isnan(epoch_loss_lst).sum()
    if num_nans > 0:
        print(f'Epoch {epoch} training: {num_nans} NaN values present in {len(data_loader.dataset)} slices')

    return epoch_loss, epoch_metrics


def val_step(model, loss_func, inputs, targets):
    recons = model(inputs, targets.shape)
    step_loss = loss_func(recons, targets)
    return step_loss, recons


def make_grid_triplet(recons, targets):
    assert recons.shape == targets.shape
    if recons.size(0) > 1:
        raise NotImplementedError('Mini-batch size greater than 1 has not been implemented yet.')

    recons = recons.detach().cpu()
    targets = targets.detach().cpu()

    large = torch.max(targets)
    small = torch.min(targets)
    diff = large - small

    view_recons = (recons.clamp(min=small, max=large) - small) / diff
    view_targets = (targets - small) / diff

    view_recons = torch.squeeze(view_recons, dim=0).unsqueeze(dim=1)
    view_targets = torch.squeeze(view_targets, dim=0).unsqueeze(dim=1)

    recons_grid = torchvision.utils.make_grid(view_recons)
    targets_grid = torchvision.utils.make_grid(view_targets)

    deltas_grid = targets_grid - recons_grid

    return recons_grid, targets_grid, deltas_grid


def val_epoch(model, loss_func, data_loader, writer, device, epoch, max_imgs=0, verbose=True, metrics=None):
    model.eval()
    torch.autograd.set_grad_enabled(False)

    epoch_loss_lst = list()
    epoch_metrics_lst = [list() for _ in metrics] if metrics else None

    for step, (inputs, targets) in enumerate(data_loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device, non_blocking=True)
        step_loss, recons = val_step(model, loss_func, inputs, targets)

        with torch.no_grad():  # Probably not actually necessary...
            epoch_loss_lst.append(step_loss.item())
            if metrics:
                step_metrics = [metric(recons, targets) for metric in metrics]
                for step_metric, epoch_metric_lst in zip(step_metrics, epoch_metrics_lst):
                    epoch_metric_lst.append(step_metric.item())  # Equation for running average.

            if verbose:
                print(f'Validation loss Epoch {epoch:03d} Step {step:03d}: {step_loss.item():.4e}')
                if metrics:
                    for step_metric in step_metrics:
                        print(f'Validation metric Epoch {epoch:03d} Step {step:03d}: {step_metric.item():.4e}')

            if max_imgs:
                interval = len(data_loader.dataset) // max_imgs
                if step % interval == 0:  # Note that all images are scaled independently of all other images.

                    assert isinstance(writer, SummaryWriter)

                    recons_grid, targets_grid, deltas_grid = make_grid_triplet(recons, targets)
                    writer.add_image('Recons', recons_grid, epoch)
                    writer.add_image('Targets', targets_grid, epoch)
                    writer.add_image('Deltas', deltas_grid, epoch)

    epoch_loss = np.nanmean(epoch_loss_lst)  # Remove nan values just in case.
    epoch_metrics = [np.nanmean(epoch_metric_lst) for epoch_metric_lst in epoch_metrics_lst] if metrics else None

    num_nans = np.isnan(epoch_loss_lst).sum()
    if num_nans > 0:
        print(f'Epoch {epoch} validation: {num_nans} NaN values present in {len(data_loader.dataset)} slices')

    return epoch_loss, epoch_metrics


def train_model(args):

    if args.batch_size > 1:
        raise NotImplementedError('Only batch size of 1 for now.')

    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Create Datasets. Use one slice at a time for now.
    train_loader, val_loader = create_data_loaders(args)

    # Define model.
    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
    # data_chans indicates the number of channels in the data.
    model = UnetModel(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
                      num_pool_layers=args.num_pool_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    loss_func = nn.MSELoss(reduction='mean').to(device)
    metrics = None

    checkpointer = CheckpointManager(model=model, optimizer=optimizer, mode='min', save_best_only=args.save_best_only,
                                     ckpt_dir=ckpt_path, max_to_keep=args.max_to_keep)

    if hasattr(args, 'previous_model') and args.previous_model:
        checkpointer.load(load_dir=checkpointer, load_optimizer=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True, cooldown=0, min_lr=1E-7)

    writer = SummaryWriter(log_dir=str(log_path))

    # example_inputs = torch.ones(size=(1, 2, 640, 328)).to(device, non_blocking=True)
    # writer.add_graph(model=model, input_to_model=example_inputs)

    logger.info('Beginning Training loop')
    for epoch in range(1, args.num_epochs + 1):  # 1 based indexing
        # Training
        tic = time()
        train_epoch_loss, train_epoch_metrics = train_epoch(
            model=model, optimizer=optimizer, loss_func=loss_func, data_loader=train_loader, device=device,
            epoch=epoch, verbose=args.verbose, metrics=metrics)

        toc = int(time() - tic)
        logger.info(f'Epoch {epoch:03d} Training. loss: {train_epoch_loss:.4e}, Time: {toc // 60}min {toc % 60}sec')
        writer.add_scalar('train_epoch_loss', scalar_value=train_epoch_loss, global_step=epoch)

        if isinstance(train_epoch_metrics, list):  # The metrics being returned are either 'None' or a list of values.
            for idx, train_epoch_metric in enumerate(train_epoch_metrics, start=1):
                logger.info(f'Epoch {epoch:03d} Training. Metric {idx}: {train_epoch_metric}')
                writer.add_scalar(f'train_epoch_metric_{idx}', scalar_value=train_epoch_metric, global_step=epoch)

        # Evaluating
        tic = time()
        val_epoch_loss, val_epoch_metrics = val_epoch(
            model=model, loss_func=loss_func, data_loader=val_loader, writer=writer, device=device,
            epoch=epoch, max_imgs=args.max_imgs, verbose=args.verbose, metrics=metrics)

        toc = int(time() - tic)
        logger.info(f'Epoch {epoch:03d} Validation. loss: {val_epoch_loss:.4e}, Time: {toc // 60}min {toc % 60}sec')
        writer.add_scalar('val_epoch_loss', scalar_value=val_epoch_loss, global_step=epoch)
        if isinstance(val_epoch_metrics, list):  # The metrics being returned are either 'None' or a list of values.
            for idx, val_epoch_metric in enumerate(val_epoch_metrics, start=1):
                logger.info(f'Epoch {epoch:03d} Validation. Metric {idx}: {val_epoch_metric}')
                writer.add_scalar(f'val_epoch_metric_{idx}', scalar_value=val_epoch_metric, global_step=epoch)

        for idx, group in enumerate(optimizer.param_groups, start=1):
            writer.add_scalar(f'learning_rate_{idx}', group['lr'], epoch)

        checkpointer.save(metric=val_epoch_loss, verbose=True)

        scheduler.step(metrics=val_epoch_loss)

    return model
