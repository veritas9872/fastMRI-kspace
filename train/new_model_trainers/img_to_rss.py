import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm

from time import time
from collections import defaultdict

from utils.run_utils import get_logger
from utils.train_utils import CheckpointManager, standardize_image
from metrics.new_1d_ssim import SSIM
from metrics.custom_losses import psnr, nmse


# Send this somewhere else soon...
def get_class_name(obj):
    return 'None' if obj is None else str(obj.__class__).split("'")[1]


class ModelTrainerRSS:
    """
    Model trainer that has RSS outputs. The inputs maybe complex or magnitude images.
    """
    def __init__(self, args, model, optimizer, train_loader, val_loader, input_train_transform, input_val_transform,
                 output_train_transform, output_val_transform, losses, scheduler=None):

        # Allow multiple processes to access tensors on GPU. Add checking for multiple continuous runs.
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method(method='spawn')

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(optimizer, optim.Optimizer), '`optimizer` must be a Pytorch Optimizer.'
        assert isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader), \
            '`train_loader` and `val_loader` must be Pytorch DataLoader objects.'

        assert callable(input_train_transform) and callable(input_val_transform), \
            'input_transforms must be callable functions.'
        # I think this would be best practice.
        assert isinstance(output_train_transform, nn.Module) and isinstance(output_val_transform, nn.Module), \
            '`output_train_transform` and `output_val_transform` must be Pytorch Modules.'

        # 'losses' is expected to be a dictionary.
        # Even composite losses should be a single loss module with a tuple as its output.
        losses = nn.ModuleDict(losses)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_scheduler = True
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                self.metric_scheduler = False
            else:
                raise TypeError('`scheduler` must be a Pytorch Learning Rate Scheduler.')

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.max_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(val_loader.dataset) // (args.max_images * args.batch_size))

        self.manager = CheckpointManager(model, optimizer, mode='min', save_best_only=args.save_best_only,
                                         ckpt_dir=args.ckpt_path, max_to_keep=args.max_to_keep)

        # loading from checkpoint if specified.
        if vars(args).get('prev_model_ckpt'):
            self.manager.load(load_dir=args.prev_model_ckpt, load_optimizer=False)

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_train_transform = input_train_transform
        self.input_val_transform = input_val_transform
        self.output_train_transform = output_train_transform
        self.output_val_transform = output_val_transform
        self.losses = losses
        self.scheduler = scheduler
        self.writer = SummaryWriter(str(args.log_path))

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.smoothing_factor = args.smoothing_factor
        self.use_slice_metrics = args.use_slice_metrics

        # This part should get SSIM, not 1 - SSIM.
        self.ssim = SSIM(filter_size=7).to(device=args.device)  # Needed to cache the kernel.

        # Logging all components of the Model Trainer.
        # Train and Val input and output transforms are assumed to use the same input transform class.
        self.logger.info(f'''
        Summary of Model Trainer Components:
        Model: {get_class_name(model)}.
        Optimizer: {get_class_name(optimizer)}.
        Input Transforms: {get_class_name(input_val_transform)}.
        Output Transform: {get_class_name(output_val_transform)}.
        RSS Image Domain Loss: {get_class_name(losses['rss_loss'])}.
        Learning-Rate Scheduler: {get_class_name(scheduler)}.
        ''')  # This part has parts different for IMG and CMG losses!!

    def train_model(self):
        tic_tic = time()
        self.logger.info('Beginning Training Loop.')
        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing of epochs.
            tic = time()  # Training
            train_epoch_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, train_epoch_loss, train_epoch_metrics, elapsed_secs=toc, training=True)

            tic = time()  # Validation
            val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, val_epoch_loss, val_epoch_metrics, elapsed_secs=toc, training=False)

            self.manager.save(metric=val_epoch_loss, verbose=True)

            if self.scheduler is not None:
                if self.metric_scheduler:  # If the scheduler is a metric based scheduler, include metrics.
                    self.scheduler.step(metrics=val_epoch_loss)
                else:
                    self.scheduler.step()

        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def train_model_(self, train_ratio: int):
        tic_tic = time()
        self.logger.info('Beginning Asymmetric Training Loop.')

        val_epoch_loss = 2  # Hack...

        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing of epochs.
            tic = time()  # Training
            train_epoch_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, train_epoch_loss, train_epoch_metrics, elapsed_secs=toc, training=True)

            if epoch % train_ratio == 0:
                tic = time()  # Validation
                val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch=epoch)
                toc = int(time() - tic)
                self._log_epoch_outputs(epoch, val_epoch_loss, val_epoch_metrics, elapsed_secs=toc, training=False)

            self.manager.save(metric=val_epoch_loss, verbose=True)

            if self.scheduler is not None:
                if self.metric_scheduler:  # If the scheduler is a metric based scheduler, include metrics.
                    self.scheduler.step(metrics=val_epoch_loss)
                else:
                    self.scheduler.step()

        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def _train_epoch(self, epoch):
        self.model.train()
        torch.autograd.set_grad_enabled(True)

        epoch_loss = list()  # Appending values to list due to numerical underflow and NaN values.
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.train_loader, start=1)
        if not self.verbose:  # tqdm has to be on the outermost iterator to function properly.
            # Known but minor bug: The tqdm total is accurate only when batch size is 1.
            data_loader = tqdm(data_loader, total=len(self.train_loader.dataset))

        for step, data in data_loader:
            # Data pre-processing is expected to have gradient calculations removed inside already.
            inputs, targets, extra_params = self.input_train_transform(*data)

            # 'recons' is a dictionary containing k-space, complex image, and real image reconstructions.
            recons, step_loss, step_metrics = self._train_step(inputs, targets, extra_params)
            epoch_loss.append(step_loss.detach())  # Perhaps not elegant, but underflow makes this necessary.

            # Gradients are not calculated so as to boost speed and remove weird errors.
            with torch.no_grad():  # Update epoch loss and metrics
                if self.use_slice_metrics:
                    slice_metrics = self._get_slice_metrics(recons, targets, extra_params)
                    step_metrics.update(slice_metrics)

                [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

                if self.verbose:
                    self._log_step_outputs(epoch, step, step_loss, step_metrics, training=True)

        # Converted to scalar and dict with scalar values respectively.
        return self._get_epoch_outputs(epoch, epoch_loss, epoch_metrics, training=True)

    def _train_step(self, inputs, targets, extra_params):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        recons = self.output_train_transform(outputs, targets, extra_params)
        step_loss, step_metrics = self._step(recons, targets, extra_params)
        step_loss.backward()
        self.optimizer.step()
        return recons, step_loss, step_metrics

    def _val_epoch(self, epoch):
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss = list()
        epoch_metrics = defaultdict(list)

        # 1 based indexing for steps.
        data_loader = enumerate(self.val_loader, start=1)
        if not self.verbose:
            data_loader = tqdm(data_loader, total=len(self.val_loader.dataset))

        for step, data in data_loader:
            inputs, targets, extra_params = self.input_val_transform(*data)
            recons, step_loss, step_metrics = self._val_step(inputs, targets, extra_params)
            epoch_loss.append(step_loss.detach())

            if self.use_slice_metrics:
                slice_metrics = self._get_slice_metrics(recons, targets, extra_params)
                step_metrics.update(slice_metrics)

            [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

            if self.verbose:
                self._log_step_outputs(epoch, step, step_loss, step_metrics, training=False)

            # Visualize images on TensorBoard.
            self._visualize_images(recons, targets, extra_params, epoch, step, training=False)

        # Converted to scalar and dict with scalar values respectively.
        return self._get_epoch_outputs(epoch, epoch_loss, epoch_metrics, training=False)

    def _val_step(self, inputs, targets, extra_params):
        outputs = self.model(inputs)
        recons = self.output_val_transform(outputs, targets, extra_params)
        step_loss, step_metrics = self._step(recons, targets, extra_params)
        return recons, step_loss, step_metrics

    def _step(self, recons, targets, extra_params):
        step_loss = self.losses['rss_loss'](recons['rss_recons'], targets['rss_targets'])

        # If step_loss is a tuple, it is expected to contain all its component losses as a dict in its second element.
        rss_metrics = dict()
        step_metrics = dict()
        if isinstance(step_loss, tuple):
            step_loss, rss_metrics = step_loss

        if 'acceleration' in extra_params:  # Different metrics for different accelerations.
            acc = extra_params['acceleration']
            if rss_metrics:  # This has to be checked before anything is added to step_metrics.
                for key, value in rss_metrics.items():
                    step_metrics[f'acc_{acc}_{key}'] = value
            step_metrics[f'acc_{acc}_loss'] = step_loss
            step_metrics.update(rss_metrics)

        return step_loss, step_metrics

    def _visualize_images(self, recons, targets, extra_params, epoch, step, training=False):
        mode = 'Training' if training else 'Validation'

        # This numbering scheme seems to have issues for certain numbers.
        # Please check cases when there is no remainder.
        if self.display_interval and (step % self.display_interval == 0):

            acc = extra_params['acceleration']
            kwargs = dict(global_step=epoch, dataformats='HW')

            # Adding RSS images of reconstructions and targets.
            recon_rss = standardize_image(recons['rss_recons'])
            delta_rss = standardize_image(targets['rss_targets'] - recons['rss_recons'])
            self.writer.add_image(f'{mode} RSS Recons/{acc}/{step}', recon_rss, **kwargs)
            self.writer.add_image(f'{mode} RSS Deltas/{acc}/{step}', delta_rss, **kwargs)

            if epoch == 1:  # Maybe add input images too later on.
                # Not actually the input but the RSS of the input images.
                input_rss = standardize_image(targets['rss_inputs'])
                target_rss = standardize_image(targets['rss_targets'])
                self.writer.add_image(f'{mode} RSS Inputs/{acc}/{step}', input_rss, **kwargs)
                self.writer.add_image(f'{mode} RSS Targets/{acc}/{step}', target_rss, **kwargs)

    def _get_slice_metrics(self, recons, targets, extra_params):
        rss_metrics = dict()
        rss_recons = recons['rss_recons'].detach()
        rss_targets = targets['rss_targets'].detach()

        rss_ssim = self.ssim(rss_recons, rss_targets)
        rss_psnr = psnr(rss_recons, rss_targets)
        rss_nmse = nmse(rss_recons, rss_targets)

        rss_metrics['rss/ssim'] = rss_ssim
        rss_metrics['rss/psnr'] = rss_psnr
        rss_metrics['rss/nmse'] = rss_nmse

        # Additional metrics for separating between acceleration factors.
        acc = extra_params["acceleration"]
        rss_metrics[f'rss_acc_{acc}/ssim'] = rss_ssim
        rss_metrics[f'rss_acc_{acc}/psnr'] = rss_psnr
        rss_metrics[f'rss_acc_{acc}/nmse'] = rss_nmse

        return rss_metrics

    def _get_epoch_outputs(self, epoch, epoch_loss, epoch_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        epoch_loss = torch.stack(epoch_loss)
        is_finite = torch.isfinite(epoch_loss)
        num_nans = (is_finite.size(0) - is_finite.sum()).item()

        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices.'
                                f'Turning on anomaly detection.')
            # Turn on anomaly detection for finding where the nan values are.
            torch.autograd.set_detect_anomaly(True)
            epoch_loss = torch.mean(epoch_loss[is_finite]).item()
        else:
            epoch_loss = torch.mean(epoch_loss).item()

        for key, value in epoch_metrics.items():
            epoch_metric = torch.stack(value)
            is_finite = torch.isfinite(epoch_metric)
            num_nans = (is_finite.size(0) - is_finite.sum()).item()

            if num_nans > 0:
                self.logger.warning(f'Epoch {epoch} {mode} {key}: {num_nans} NaN values present in {num_slices} slices.'
                                    f'Turning on anomaly detection.')
                epoch_metrics[key] = torch.mean(epoch_metric[is_finite]).item()
            else:
                epoch_metrics[key] = torch.mean(epoch_metric).item()

        return epoch_loss, epoch_metrics

    def _log_step_outputs(self, epoch, step, step_loss, step_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_loss.item():.4e}')
        for key, value in step_metrics.items():
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d}: {mode} {key}: {value.item():.4e}')

    def _log_epoch_outputs(self, epoch, epoch_loss, epoch_metrics, elapsed_secs, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} {mode}. loss: {epoch_loss:.4e}, '
                         f'Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode} epoch_loss', scalar_value=epoch_loss, global_step=epoch)

        for key, value in epoch_metrics.items():
            self.logger.info(f'Epoch {epoch:03d} {mode}. {key}: {value:.4e}')
            # Very important whether it is mode_~~ or mode/~~.
            if 'loss' in key:
                self.writer.add_scalar(f'{mode}/epoch_{key}', scalar_value=value, global_step=epoch)
            else:
                self.writer.add_scalar(f'{mode}_epoch_{key}', scalar_value=value, global_step=epoch)

        if not training:  # Record learning rate.
            for idx, group in enumerate(self.optimizer.param_groups, start=1):
                self.writer.add_scalar(f'learning_rate_{idx}', group['lr'], global_step=epoch)
