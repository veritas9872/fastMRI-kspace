import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from time import time
from collections.abc import Iterable

from utils.run_utils import get_logger
from utils.train_utils import CheckpointManager, make_grid_triplet, make_k_grid
from data.data_transforms import complex_abs, fft2


class ModelTrainerC2C:
    """
    Model trainer for Complex Image Learning.
    """
    def __init__(self, args, model, optimizer, train_loader, val_loader,
                 post_processing, c_loss, metrics=None, scheduler=None):

        multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(optimizer, optim.Optimizer), '`optimizer` must be a Pytorch Optimizer.'
        assert isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader), \
            '`train_loader` and `val_loader` must be Pytorch DataLoader objects.'

        # I think this would be best practice.
        assert isinstance(post_processing, nn.Module), '`post_processing` must be a Pytorch Module.'

        # This is not a mistake. Pytorch implements loss functions as modules.
        assert isinstance(c_loss, nn.Module), '`c_loss` must be a callable Pytorch Module.'

        if metrics is not None:
            assert isinstance(metrics, Iterable), '`metrics` must be an iterable, preferably a list or tuple.'
            for metric in metrics:
                assert callable(metric), 'All metrics must be callable functions.'

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_scheduler = True
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                self.metric_scheduler = False
            else:
                raise TypeError('`scheduler` must be a Pytorch Learning Rate Scheduler.')

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.post_processing_func = post_processing
        self.c_loss_func = c_loss
        self.metrics = metrics
        self.scheduler = scheduler

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.writer = SummaryWriter(logdir=str(args.log_path))

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.max_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(self.val_loader.dataset) // (args.max_images * args.batch_size))

        self.checkpointer = CheckpointManager(
            model=self.model, optimizer=self.optimizer, mode='min', save_best_only=args.save_best_only,
            ckpt_dir=args.ckpt_path, max_to_keep=args.max_to_keep)

        # loading from checkpoint if specified.
        if vars(args).get('prev_model_ckpt'):
            self.checkpointer.load(load_dir=args.prev_model_ckpt, load_optimizer=False)

    def train_model(self):
        tic_tic = time()
        self.logger.info('Beginning Training Loop.')
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

            if self.scheduler is not None:
                if self.metric_scheduler:  # If the scheduler is a metric based scheduler, include metrics.
                    self.scheduler.step(metrics=val_epoch_loss)
                else:
                    self.scheduler.step()

        # Finishing Training Loop
        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def _train_epoch(self, epoch):
        self.model.train()
        torch.autograd.set_grad_enabled(True)

        epoch_loss_list = list()  # Appending values to list due to numerical underflow.
        epoch_metrics_list = [list() for _ in self.metrics] if self.metrics else None

        # labels are fully sampled coil-wise images, not rss or esc.
        for step, (inputs, c_img_targets, extra_params) in enumerate(self.train_loader, start=1):
            step_loss, c_img_recons = self._train_step(inputs, c_img_targets, extra_params)

            # Gradients are not calculated so as to boost speed and remove weird errors.
            with torch.no_grad():  # Update epoch loss and metrics
                epoch_loss_list.append(step_loss.detach())  # Perhaps not elegant, but underflow makes this necessary.
                step_metrics = self._get_step_metrics(c_img_recons, c_img_targets, epoch_metrics_list)
                self._log_step_outputs(epoch, step, step_loss, step_metrics, training=True)

        epoch_loss, epoch_metrics = self._get_epoch_outputs(epoch, epoch_loss_list, epoch_metrics_list, training=True)
        return epoch_loss, epoch_metrics

    def _train_step(self, inputs, c_img_targets, extra_params):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        c_img_recons = self.post_processing_func(outputs, c_img_targets, extra_params)
        step_loss = self.c_loss_func(c_img_recons, c_img_targets)
        step_loss.backward()
        self.optimizer.step()
        return step_loss, c_img_recons

    def _val_epoch(self, epoch):
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss_list = list()
        epoch_metrics_list = [list() for _ in self.metrics] if self.metrics else None

        for step, (inputs, c_img_targets, extra_params) in enumerate(self.val_loader, start=1):
            step_loss, c_img_recons = self._val_step(inputs, c_img_targets, extra_params)

            epoch_loss_list.append(step_loss.detach())
            # Step functions have internalized conditional statements deciding whether to execute or not.
            step_metrics = self._get_step_metrics(c_img_recons, c_img_targets, epoch_metrics_list)
            self._log_step_outputs(epoch, step, step_loss, step_metrics, training=False)

            # Save images to TensorBoard.
            # Condition ensures that self.display_interval != 0 and that the step is right for display.
            if self.display_interval and (step % self.display_interval == 0):
                kspace_recons, kspace_targets, image_recons, image_targets, image_deltas \
                    = self._visualize_outputs(c_img_recons, c_img_targets, smoothing_factor=8)

                self.writer.add_image(f'k-space_Recons/{step}', kspace_recons, epoch, dataformats='HW')
                self.writer.add_image(f'k-space_Targets/{step}', kspace_targets, epoch, dataformats='HW')
                self.writer.add_image(f'Image_Recons/{step}', image_recons, epoch, dataformats='HW')
                self.writer.add_image(f'Image_Targets/{step}', image_targets, epoch, dataformats='HW')
                self.writer.add_image(f'Image_Deltas/{step}', image_deltas, epoch, dataformats='HW')

        epoch_loss, epoch_metrics = self._get_epoch_outputs(epoch, epoch_loss_list, epoch_metrics_list, training=False)
        return epoch_loss, epoch_metrics

    def _val_step(self, inputs, c_img_targets, extra_params):
        """
        All extra parameters are to be placed in extra_params.
        This makes the system more flexible.
        """
        outputs = self.model(inputs)
        c_img_recons = self.post_processing_func(outputs, c_img_targets, extra_params)
        step_loss = self.c_loss_func(c_img_recons, c_img_targets)
        return step_loss, c_img_recons

    def _get_step_metrics(self, c_img_recons, c_img_targets, epoch_metrics_list):
        if self.metrics is not None:
            step_metrics = [metric(c_img_recons, c_img_targets) for metric in self.metrics]
            for step_metric, epoch_metric_list in zip(step_metrics, epoch_metrics_list):
                epoch_metric_list.append(step_metric.detach())
            return step_metrics
        return None  # Explicitly return None for step_metrics if self.metrics is None. Not necessary but more readable.

    def _get_epoch_outputs(self, epoch, epoch_loss_list, epoch_metrics_list, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan or inf values.
        epoch_loss_tensor = torch.stack(epoch_loss_list)
        finite_values = torch.isfinite(epoch_loss_tensor)
        num_nans = len(epoch_loss_list) - int(finite_values.sum().item())
        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')
            epoch_loss = torch.mean(epoch_loss_tensor[finite_values]).item()
        else:
            epoch_loss = torch.mean(epoch_loss_tensor).item()

        if self.metrics:
            epoch_metrics = list()
            for idx, epoch_metric_list in enumerate(epoch_metrics_list, start=1):
                epoch_metric_tensor = torch.stack(epoch_metric_list)
                finite_values = torch.isfinite(epoch_metric_tensor)
                num_nans = len(epoch_metric_list) - int(finite_values.sum().item())

                if num_nans > 0:
                    self.logger.warning(
                        f'Epoch {epoch} {mode}: Metric {idx} has {num_nans} NaN values in {num_slices} slices')
                    epoch_metric = torch.mean(epoch_metric_tensor[finite_values]).item()
                else:
                    epoch_metric = torch.mean(epoch_metric_tensor).item()

                epoch_metrics.append(epoch_metric)
        else:
            epoch_metrics = None

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
            f'Epoch {epoch:03d} {mode}. loss: {epoch_loss:.4e}, Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode}_epoch_loss', scalar_value=epoch_loss, global_step=epoch)
        if isinstance(epoch_metrics, list):  # The metrics being returned are either 'None' or a list of values.
            for idx, epoch_metric in enumerate(epoch_metrics, start=1):
                self.logger.info(f'Epoch {epoch:03d} {mode}. Metric {idx}: {epoch_metric}')
                self.writer.add_scalar(f'{mode}_epoch_metric_{idx}', scalar_value=epoch_metric, global_step=epoch)

    @staticmethod
    def _visualize_outputs(c_img_recons, c_img_targets, smoothing_factor=8):
        image_recons = complex_abs(c_img_recons)
        image_targets = complex_abs(c_img_targets)
        kspace_recons = make_k_grid(fft2(c_img_recons), smoothing_factor)
        kspace_targets = make_k_grid(fft2(c_img_targets), smoothing_factor)
        image_recons, image_targets, image_deltas = make_grid_triplet(image_recons, image_targets)
        return kspace_recons, kspace_targets, image_recons, image_targets, image_deltas
