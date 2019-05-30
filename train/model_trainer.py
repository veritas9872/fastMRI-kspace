import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader

import numpy as np
from tensorboardX import SummaryWriter

from time import time
from pathlib import Path

from data.data_transforms import complex_abs
from train.subsample import MaskFunc
from utils.train_utils import CheckpointManager
from utils.run_utils import get_logger
from data.mri_data import SliceData
from data.pre_processing import KInputSliceTransform


"""
Please note a bit of terminology. 
In this file, 'recons' indicates coil-wise reconstructions,
not final reconstructions for submissions.
Also, 'targets' indicates coil-wise targets, not the 320x320 ground-truth labels.
k-slice means a slice of k-space, i.e. only 1 slice of k-space.
"""


# Send all these functions to train utils later.
def create_loaded_datasets(args, device):
    """
    A function for creating datasets where the data is sent to the desired device before being given to the model.
    This is done because data transfer is a serious bottleneck in k-space learning and is best done asynchronously.
    Also, the Fourier Transform is best done on the GPU instead of on CPU.
    Finally, Sending k-space data to device beforehand removes the need to also send generated label data to device.
    This reduces data transfer significantly.
    The only problem is that sending to GPU cannot be batched with this method.
    However, this seems to be a small price to pay.
    """

    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    divisor = 2 ** args.num_pool_layers  # UNET architecture requires that all inputs be dividable by some power of 2.

    # Generating Datasets.
    train_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=KInputSliceTransform(
            mask_func=train_mask_func, challenge=args.challenge, device=device, use_seed=False, divisor=divisor),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_gt=False,
        converted=args.converted
    )

    val_dataset = SliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=KInputSliceTransform(
            mask_func=val_mask_func, challenge=args.challenge, device=device, use_seed=True, divisor=divisor),
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
        pin_memory=args.pin_memory,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, val_loader


def make_grid_triplet(image_recons, targets):
    assert image_recons.size() == targets.size()
    if image_recons.size(0) > 1:
        raise NotImplementedError('Mini-batch size greater than 1 has not been implemented yet.')

    large = torch.max(targets)
    small = torch.min(targets)
    diff = large - small

    # Scaling to 0~1 range.
    image_recons = (image_recons.clamp(min=small, max=large) - small) / diff
    targets = (targets - small) / diff

    # Send to CPU if necessary. Assumes batch size of 1.
    image_recons = image_recons.detach().cpu().squeeze(dim=0)
    targets = targets.detach().cpu().squeeze(dim=0)

    if image_recons.size(0) == 15:
        image_recons = torch.cat(torch.chunk(image_recons.view(-1, image_recons.size(-1)), chunks=5, dim=0), dim=1)
        targets = torch.cat(torch.chunk(targets.view(-1, targets.size(-1)), chunks=5, dim=0), dim=1)
    elif image_recons.size(0) == 1:
        image_recons = image_recons.squeeze()
        targets = targets.squeeze()
    else:
        raise ValueError('Invalid dimensions!')

    deltas = targets - image_recons

    return image_recons, targets, deltas


def make_k_grid(kspace_recons):
    """
    Function for making k-space visualizations for Tensorboard.
    """
    if kspace_recons.size(0) > 1:
        raise NotImplementedError('Mini-batch size greater than 1 has not been implemented yet.')

    kspace_recons = complex_abs(torch.log10(kspace_recons.detach())).cpu().squeeze(dim=0)

    if kspace_recons.size(0) == 15:
        kspace_recons = torch.cat(torch.chunk(kspace_recons.view(-1, kspace_recons.size(-1)), chunks=5, dim=0), dim=1)

    kspace_recons = kspace_recons.squeeze()
    return kspace_recons


# Only this class should remain in this file.
class ModelTrainer:
    def __init__(self, args, model, optimizer, loss_func, metrics=None, scheduler=None):
        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(optimizer, optim.Optimizer), '`optimizer` must be a Pytorch Optimizer.'
        if metrics is not None:
            assert isinstance(metrics, (list, tuple)), '`metrics` must be a list or tuple.'
            for metric in metrics:
                assert callable(metric), 'All metrics must be callable functions.'

        # This is not a mistake. Pytorch implements loss functions as modules.
        assert isinstance(loss_func, nn.Module), '`loss_func` must be a callable Pytorch Module.'
        # assert callable(loss_func), '`loss_func` must be a callable function.'

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_scheduler = True
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                self.metric_scheduler = False
            else:
                raise TypeError('`scheduler` must be a Pytorch Learning Rate Scheduler.')

        self.model = model.to(args.device, non_blocking=True)
        self.optimizer = optimizer
        self.loss_func = loss_func  # I don't think it is necessary to send loss_func or metrics to device.
        self.metrics = metrics
        self.scheduler = scheduler

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.writer = SummaryWriter(log_dir=str(args.log_path))

        self.train_loader, self.val_loader = create_loaded_datasets(args=args, device=args.device)

        # Display interval of 0 means no display of validation images on TensorBoard.
        self.display_interval = int(len(self.val_loader.dataset) // args.max_images) if (args.max_images > 0) else 0

        # Writing model graph to TensorBoard. Results might not be very good.
        num_chans = 30 if args.challenge == 'multicoil' else 2
        example_inputs = torch.ones(size=(1, num_chans, 640, 328)).to(args.device)
        self.writer.add_graph(model=model, input_to_model=example_inputs)
        del example_inputs  # Remove unnecessary tensor taking up memory.

        self.checkpointer = CheckpointManager(
            model=self.model, optimizer=self.optimizer, mode='min', save_best_only=args.save_best_only,
            ckpt_dir=args.ckpt_path, max_to_keep=args.max_to_keep)

        # loading from checkpoint if specified.
        if hasattr(args, 'prev_model_ckpt') and args.prev_model_ckpt:
            self.checkpointer.load(load_dir=args.prev_model_ckpt, load_optimizer=False)

    def train_model(self):
        multiprocessing.set_start_method(method='spawn')
        self.logger.info('Beginning Training Loop.')
        tic_tic = time()
        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing
            # Training
            tic = time()
            train_epoch_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch=epoch, epoch_loss=train_epoch_loss,
                                    epoch_metrics=train_epoch_metrics, elapsed_secs=toc, training=True)

            # Validation
            tic = time()
            val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch=epoch, epoch_loss=val_epoch_loss,
                                    epoch_metrics=val_epoch_metrics, elapsed_secs=toc, training=False)

            self.checkpointer.save(metric=val_epoch_loss, verbose=True)

            if self.scheduler:
                if self.metric_scheduler:  # If the scheduler is a metric based scheduler, include metrics.
                    self.scheduler.step(metrics=val_epoch_metrics)
                else:
                    self.scheduler.step()

        # Finishing Training Loop
        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        hrs = toc_toc // 3600
        mins = toc_toc // 60
        secs = toc_toc % 60
        self.logger.info(f'Finishing Training Loop. Total elapsed time: {hrs:03d}hr {mins:02d}min {secs:02d}sec.')

    # TODO: Using the new system, post-processing will be done in the train/val steps.
    #  Also, there should be a way to set the input slice transform inside the model trainer as well.
    def _train_step(self, inputs, targets, scales):
        self.optimizer.zero_grad()
        image_recons, kspace_recons = self.model(inputs, targets, scales)
        step_loss = self.loss_func(image_recons, targets)
        step_loss.backward()
        self.optimizer.step(closure=None)  # close=None is there just to make pylint happy.
        return step_loss, image_recons, kspace_recons

    def _train_epoch(self, epoch):
        self.model.train()
        torch.autograd.set_grad_enabled(True)

        epoch_loss_lst = list()  # Appending values to list due to numerical underflow.
        epoch_metrics_lst = [list() for _ in self.metrics] if self.metrics else None

        # labels are fully sampled coil-wise images, not rss or esc.
        for step, (inputs, targets, scales) in enumerate(self.train_loader, start=1):
            step_loss, image_recons, kspace_recons = self._train_step(inputs, targets, scales)

            # Gradients are not calculated so as to boost speed and remove weird errors.
            with torch.no_grad():  # Update epoch loss and metrics
                epoch_loss_lst.append(step_loss.item())  # Perhaps not elegant, but underflow makes this necessary.

                # The step functions here have all necessary conditionals internally.
                # There is no need to externally specify whether to use them or not.
                step_metrics = self._get_step_metrics(image_recons, targets, epoch_metrics_lst)
                self._log_step_outputs(epoch, step, step_loss, step_metrics, training=True)

        epoch_loss, epoch_metrics = self._get_epoch_outputs(epoch, epoch_loss_lst, epoch_metrics_lst, training=True)
        return epoch_loss, epoch_metrics

    def _val_step(self, inputs, targets, scales):
        """
        This implementation assumes that model outputs are Tensors, not lists.
        It also assumes that scalar multiplication was the only processing step, allowing for the possibility
        that each slice had a different scalar multiplied to it (e.g. the pseudo-std normalization).
        'scales' indicates multiple 'scaling' values from the data pre-processing step.
        This works because the FFT and IFFT are linear functions.
        It would be most efficient to multiply the scaling just once inside the model before the IFFT2D.
        However, this would make the code too dirty.  --> Just do it anyway.
        """

        # TODO: Implement post-processing here.
        image_recons, kspace_recons = self.model(inputs, targets, scales)
        step_loss = self.loss_func(image_recons, targets)
        return step_loss, image_recons, kspace_recons

    def _val_epoch(self, epoch):
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss_lst = list()
        epoch_metrics_lst = [list() for _ in self.metrics] if self.metrics else None

        for step, (inputs, targets, scales) in enumerate(self.val_loader, start=1):
            step_loss, image_recons, kspace_recons = self._val_step(inputs, targets, scales)

            epoch_loss_lst.append(step_loss.item())
            # Step functions have internalized conditional statements deciding whether to execute or not.
            step_metrics = self._get_step_metrics(image_recons, targets, epoch_metrics_lst)
            self._log_step_outputs(epoch, step, step_loss, step_metrics, training=False)

            # Save images to TensorBoard. Send this to a separate function later on.
            # Condition ensures that self.display_interval != 0 and that the step is right for display.
            if self.display_interval and (step % self.display_interval == 0):
                recons_grid, targets_grid, deltas_grid = make_grid_triplet(image_recons, targets)
                kspace_grid = make_k_grid(kspace_recons)

                self.writer.add_image(f'k-space_Recons/{step}', kspace_grid, epoch, dataformats='HW')
                self.writer.add_image(f'Image_Recons/{step}', recons_grid, epoch, dataformats='HW')
                self.writer.add_image(f'Targets/{step}', targets_grid, epoch, dataformats='HW')
                self.writer.add_image(f'Deltas/{step}', deltas_grid, epoch, dataformats='HW')

        epoch_loss, epoch_metrics = self._get_epoch_outputs(epoch, epoch_loss_lst, epoch_metrics_lst, training=False)
        return epoch_loss, epoch_metrics

    def _get_step_metrics(self, image_recons, targets, epoch_metrics_lst):
        if self.metrics is not None:
            step_metrics = [metric(image_recons, targets) for metric in self.metrics]
            for step_metric, epoch_metric_lst in zip(step_metrics, epoch_metrics_lst):
                epoch_metric_lst.append(step_metric.item())
            return step_metrics
        return None  # Explicitly return None for step_metrics when self.metrics is None.

    def _get_epoch_outputs(self, epoch, epoch_loss_lst, epoch_metrics_lst, training=True):
        mode = 'training' if training else 'validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        num_nans = np.isnan(epoch_loss_lst).sum()
        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')

        epoch_loss = float(np.nanmean(epoch_loss_lst))  # Remove nan values just in case.
        epoch_metrics = [float(np.nanmean(epoch_metric_lst)) for epoch_metric_lst in
                         epoch_metrics_lst] if self.metrics else None

        return epoch_loss, epoch_metrics

    def _log_step_outputs(self, epoch, step, step_loss, step_metrics, training=True):
        if self.verbose:
            mode = 'Training' if training else 'Validation'
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_loss.item():.4e}')
            if self.metrics:
                for idx, step_metric in enumerate(step_metrics):
                    self.logger.info(
                        f'Epoch {epoch:03d} Step {step:03d}: {mode} metric {idx}: {step_metric.item():.4e}')

    def _log_epoch_outputs(self, epoch, epoch_loss, epoch_metrics, elapsed_secs, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(
            f'Epoch {epoch:03d} {mode}. loss: {epoch_loss:.4e}, Time: {elapsed_secs // 60}min {elapsed_secs % 60}sec')
        self.writer.add_scalar(f'{mode}_epoch_loss', scalar_value=epoch_loss, global_step=epoch)
        if isinstance(epoch_metrics, list):  # The metrics being returned are either 'None' or a list of values.
            for idx, epoch_metric in enumerate(epoch_metrics, start=1):
                self.logger.info(f'Epoch {epoch:03d} {mode}. Metric {idx}: {epoch_metric}')
                self.writer.add_scalar(f'{mode}_epoch_metric_{idx}', scalar_value=epoch_metric, global_step=epoch)
