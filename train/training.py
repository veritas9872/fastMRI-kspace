import torch
import torchvision
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

from pathlib import Path
from time import time

from utils.run_utils import get_logger, initialize, save_dict_as_json, create_arg_parser
from utils.train_utils import CheckpointManager
from train.subsample import MaskFunc
from data.mri_data import SliceData
from data.data_transforms import DataTrainTransform
from eda.k_unet_mode_test import UnetModel


"""
Please note a bit of terminology. 
In this file, 'recons' indicate coil-wise reconstructions,
not final reconstructions for submissions.
"""


def create_datasets(args):
    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=DataTrainTransform(train_mask_func, args.challenge, use_seed=False),
        challenge=args.challenge,
        sample_rate=args.sample_rate
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=DataTrainTransform(val_mask_func, args.challenge, use_seed=True),
        challenge=args.challenge,
        sample_rate=args.sample_rate
    )

    return train_dataset, val_dataset


def create_data_loaders(args):
    train_dataset, val_dataset = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_step(model, optimizer, loss_func, images, labels):
    optimizer.zero_grad()
    recons = model(images, labels.shape)
    step_loss = loss_func(recons, labels)  # Pytorch uses (input, target) ordering.
    step_loss.backward()
    optimizer.step()
    return step_loss, recons


def train_epoch(model, optimizer, loss_func, data_loader, device, epoch, verbose=True, metrics=None):
    model.train()
    torch.autograd.set_grad_enabled(True)

    # initialize epoch loss
    epoch_loss = 0.  # Automatically converts to pytorch tensor.
    # initialize multiple epoch metrics
    epoch_metrics = [0. for _ in metrics] if metrics else None

    # labels are fully sampled coil-wise images, not rss or esc.
    for idx, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device)
        labels = labels.to(device, non_blocking=True)
        step_loss, recons = train_step(model, optimizer, loss_func, images, labels)

        # Gradients are not calculated to boost speed and remove weird errors.
        with torch.no_grad():  # Update epoch loss and metrics
            epoch_loss += (step_loss.item() - epoch_loss) / idx  # Equation for running average.
            if metrics:
                step_metrics = [metric(recons, labels) for metric in metrics]
                for step_metric, epoch_metric in zip(step_metrics, epoch_metrics):
                    epoch_metric += (step_metric.item() - epoch_metric) / idx

            if verbose:
                print(f'Training loss Epoch {epoch:03d} Step {idx:03d}: {step_loss.item():.4e}')
                if metrics:
                    for step_metric in step_metrics:
                        print(f'Training metric Epoch {epoch:03d} Step {idx:03d}: {step_metric.item():.4e}')

    return epoch_loss, epoch_metrics


def val_step(model, loss_func, images, labels):
    recons = model(images, labels.shape)
    step_loss = loss_func(recons, labels)
    return step_loss, recons


def make_grid_for_one_image(recons):  # Helper for saving to TensorboardX or as image
    if recons.size(0) > 1:  # Singlecoil is not implemented either
        raise NotImplementedError('Mini-batch size greater than 1 has not been implemented yet.')

    # Note that each image scales independently of all other images. This may cause weird scaling behavior.
    grid = torch.squeeze(recons).unsqueeze(dim=1)  # Assumes batch_size=1
    # Weird bug where nrow parameter seems to decide number of columns instead of rows.
    grid = torchvision.utils.make_grid(grid, nrow=5, normalize=True, scale_each=True, pad_value=1.)
    return np.squeeze(grid.cpu().numpy())  # Since there should only be 1 channel.


def val_epoch(model, loss_func, data_loader, writer, device, epoch, max_imgs=0, verbose=True, metrics=None):
    model.eval()
    torch.autograd.set_grad_enabled(False)

    epoch_loss = 0.
    epoch_metrics = [0. for _ in metrics] if metrics else None

    for step, (data, targets) in enumerate(data_loader, start=1):
        data = data.to(device)
        targets = targets.to(device, non_blocking=True)
        step_loss, recons = val_step(model, loss_func, data, targets)

        with torch.no_grad():  # Probably not actually necessary...
            epoch_loss += (step_loss.item() - epoch_loss) / step  # Equation for running average.
            if metrics:
                step_metrics = [metric(recons, targets) for metric in metrics]
                for step_metric, epoch_metric in zip(step_metrics, epoch_metrics):
                    epoch_metric += (step_metric.item() - epoch_metric) / step

            if verbose:
                print(f'Validation loss Epoch {epoch:03d} Step {step:03d}: {step_loss.item():.4e}')
                if metrics:
                    for step_metric in step_metrics:
                        print(f'Validation metric Epoch {epoch:03d} Step {step:03d}: {step_metric.item():.4e}')

            if max_imgs:
                interval = len(data_loader.dataset) // max_imgs
                if step % interval == 0:  # Note that all images are scaled independently of all other images.

                    recon_grid = make_grid_for_one_image(recons)
                    assert isinstance(writer, SummaryWriter)
                    print(recon_grid.shape)
                    writer.add_image('Recons', recon_grid, epoch)

                    target_grid = make_grid_for_one_image(targets)
                    writer.add_image('Targets', target_grid, epoch)

                    delta_grid = make_grid_for_one_image(targets - recons)
                    writer.add_image('Delta', delta_grid, epoch)

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
    # TODO: I must verify whether the output is correct. The image outputs look like k-space right now.
    model = UnetModel(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
                      num_pool_layers=args.num_pool_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    loss_func = nn.L1Loss(reduction='mean').to(device)

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
            epoch=epoch, verbose=args.verbose, metrics=None)

        toc = int(time() - tic)
        logger.info(f'Epoch {epoch:03d} Training. loss: {train_epoch_loss:.4f}, Time: {toc // 60}min {toc % 60}sec')
        writer.add_scalar('train_epoch_loss', scalar_value=train_epoch_loss, global_step=epoch)

        # Evaluating
        tic = time()
        val_epoch_loss, val_epoch_metrics = val_epoch(
            model=model, loss_func=loss_func, data_loader=val_loader, writer=writer, device=device,
            epoch=epoch, max_imgs=args.max_imgs, verbose=args.verbose, metrics=None)

        toc = int(time() - tic)
        logger.info(f'Epoch {epoch:03d} Validation. loss: {val_epoch_loss:.4f}, Time: {toc // 60}min {toc % 60}sec')
        writer.add_scalar('val_epoch_loss', scalar_value=val_epoch_loss, global_step=epoch)

        for idx, group in enumerate(optimizer.param_groups, start=1):
            writer.add_scalar(f'learning_rate_{idx}', group['lr'], epoch)

        checkpointer.save(metric=val_epoch_loss, verbose=True)

        scheduler.step(metrics=val_epoch_loss)



