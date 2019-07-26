import torch
from torch import nn, optim, multiprocessing
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from time import time
from collections import defaultdict

from utils.run_utils import get_logger
from utils.train_utils import CheckpointManager, make_grid_triplet, make_k_grid, make_input_triplet, \
                            make_input_RSS, make_RSS
from metrics.my_ssim import ssim_loss
from metrics.custom_losses import psnr_loss, nmse_loss

from data.data_transforms import root_sum_of_squares, pre_RSS


class ModelTrainerIMG:
    """
    Model Trainer for k-space learning or complex image learning
    with losses in complex image domains and real valued image domains.
    All learning occurs in k-space or complex image domains
    while all losses are obtained from either complex images or real-valued images.
    """

    def __init__(self, args, model, optimizer, train_loader, val_loader,
                 input_train_transform, input_val_transform, output_transform, losses, scheduler=None):

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
        assert isinstance(output_transform, nn.Module), '`output_transform` must be a Pytorch Module.'

        # 'losses' is expected to be a dictionary.
        losses = nn.ModuleDict(losses)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_scheduler = True
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                self.metric_scheduler = False
            else:
                raise TypeError('`scheduler` must be a Pytorch Learning Rate Scheduler.')

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.display_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(val_loader.dataset) // (args.display_images * args.batch_size))

        self.checkpointer = CheckpointManager(model, optimizer, mode='min', save_best_only=args.save_best_only,
                                              ckpt_dir=args.ckpt_path, max_to_keep=args.max_to_keep)

        # loading from checkpoint if specified.
        if vars(args).get('prev_model_ckpt'):
            self.checkpointer.load(load_dir=args.prev_model_ckpt, load_optimizer=False)

        self.name = args.name
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_train_transform = input_train_transform
        self.input_val_transform = input_val_transform
        self.output_transform = output_transform
        self.losses = losses
        self.scheduler = scheduler

        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.smoothing_factor = args.smoothing_factor
        self.use_slice_metrics = args.use_slice_metrics
        self.img_lambda1 = torch.tensor(args.img_lambda1, dtype=torch.float32, device=args.device)
        self.img_lambda2 = torch.tensor(args.img_lambda2, dtype=torch.float32, device=args.device)
        self.writer = SummaryWriter(str(args.log_path))

    def train_model(self):
        tic_tic = time()
        self.logger.info(self.name)
        self.logger.info('Beginning Training Loop.')
        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing
            # Training
            tic = time()
            train_epoch_loss, train_epoch_metrics = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, train_epoch_loss, train_epoch_metrics,
                                    elapsed_secs=toc, training=True, verbose=True)

            # Validation
            tic = time()
            val_epoch_loss, val_epoch_metrics = self._val_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch, val_epoch_loss, val_epoch_metrics,
                                    elapsed_secs=toc, training=False, verbose=True)

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

        epoch_loss = list()  # Appending values to list due to numerical underflow.
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.train_loader, start=1)
        if not self.verbose:  # tqdm has to be on the outermost iterator to function properly.
            data_loader = tqdm(data_loader, total=len(self.train_loader.dataset))

        # 'targets' is a dictionary containing k-space targets, cmg_targets, and img_targets.
        for step, data in data_loader:
            with torch.no_grad():  # Data pre-processing should be done without gradients.
                inputs, targets, extra_params = self.input_train_transform(*data)

            # 'recons' is a dictionary containing k-space, complex image, and real image reconstructions.
            recons, step_loss, step_metrics = self._train_step(inputs, targets, extra_params)
            epoch_loss.append(step_loss.detach())  # Perhaps not elegant, but underflow makes this necessary.

            # Gradients are not calculated so as to boost speed and remove weird errors.
            with torch.no_grad():  # Update epoch loss and metrics
                if self.use_slice_metrics:
                    slice_metrics = self._get_slice_metrics(recons['img_recons'], targets['img_targets'])
                    step_metrics.update(slice_metrics)
                [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

                if self.verbose:
                    self._log_step_outputs(epoch, step, step_loss, step_metrics, training=True)

        # Converted to scalar and dict with scalar forms.
        return self._get_epoch_outputs(epoch, epoch_loss, epoch_metrics, training=True)

    def _train_step(self, inputs, targets, extra_params):
        self.optimizer.zero_grad()
        outputs = self.model(inputs, extra_params['mask_holder'])
        recons = self.output_transform(outputs, targets, extra_params)

        # Expects a single loss. No loss decomposition within domain implemented yet.
        cmg_loss = self.losses['cmg_loss'](recons['cmg_recons'], targets['cmg_targets'])
        img_loss1 = self.losses['img_loss1'](recons['img_recons'], targets['img_targets'])
        img_loss2 = self.losses['img_loss2'](recons['img_recons'], targets['img_targets'])
        step_loss = cmg_loss + self.img_lambda1 * img_loss1 + self.img_lambda2 * img_loss2
        step_loss.backward()
        self.optimizer.step()
        step_metrics = {'cmg_loss': cmg_loss, 'img_loss1': img_loss1, 'img_loss2': img_loss2}
        return recons, step_loss, step_metrics

    def _val_epoch(self, epoch):
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss = list()
        epoch_metrics = defaultdict(list)

        data_loader = enumerate(self.val_loader, start=1)
        if not self.verbose:
            data_loader = tqdm(data_loader, total=len(self.val_loader.dataset))

        # 'targets' is a dictionary containing k-space targets, cmg_targets, and img_targets.
        for step, data in data_loader:
            inputs, targets, extra_params = self.input_val_transform(*data)
            # 'recons' is a dictionary containing k-space, complex image, and real image reconstructions.
            recons, step_loss, step_metrics = self._val_step(inputs, targets, extra_params)
            epoch_loss.append(step_loss.detach())

            if self.use_slice_metrics:
                # rss_img_recons, rss_img_targets = pre_RSS(recons['img_recons'], targets['img_targets'])
                # This RSS computation is extremely slow
                rss_img_recons = (recons['img_recons'] ** 2).sum(dim=0).sqrt()
                rss_img_targets = (targets['img_targets'] ** 2).sum(dim=0).sqrt()
                slice_metrics = self._get_slice_metrics(rss_img_recons, rss_img_targets)
                step_metrics.update(slice_metrics)

            [epoch_metrics[key].append(value.detach()) for key, value in step_metrics.items()]

            if self.verbose:
                self._log_step_outputs(epoch, step, step_loss, step_metrics, training=False)

            # Save images to TensorBoard.
            # Condition ensures that self.display_interval != 0 and that the step is right for display.
            if self.display_interval and (step % self.display_interval == 0):
                # img_recon_grid, img_target_grid, img_delta_grid = \
                #     make_grid_triplet(recons['img_recons'], targets['img_targets'])
                img_recon_grid, img_target_grid, img_delta_grid = \
                    make_RSS(recons['img_recons'], targets['img_targets'])
                # img_input_grid = make_input_triplet(extra_params['img_inputs'])
                if epoch == 1:
                    img_input_grid = make_input_RSS(extra_params['img_inputs'])
                kspace_recon_grid = make_k_grid(recons['kspace_recons'], self.smoothing_factor)
                kspace_target_grid = make_k_grid(targets['kspace_targets'], self.smoothing_factor)

                self.writer.add_image(f'k-space_Recons/{step}', kspace_recon_grid, epoch, dataformats='HW')
                self.writer.add_image(f'k-space_Targets/{step}', kspace_target_grid, epoch, dataformats='HW')
                if epoch == 1:
                    self.writer.add_image(f'Val_Image_Inputs/{step}', img_input_grid, epoch, dataformats='HW')
                self.writer.add_image(f'Val_Image_Recons/{step}', img_recon_grid, epoch, dataformats='HW')
                self.writer.add_image(f'Val_Image_Targets/{step}', img_target_grid, epoch, dataformats='HW')
                self.writer.add_image(f'Image_Deltas/{step}', img_delta_grid, epoch, dataformats='HW')

        epoch_loss, epoch_metrics = self._get_epoch_outputs(epoch, epoch_loss, epoch_metrics, training=False)
        return epoch_loss, epoch_metrics

    def _val_step(self, inputs, targets, extra_params):
        outputs = self.model(inputs, extra_params['mask_holder'])
        recons = self.output_transform(outputs, targets, extra_params)
        cmg_loss = self.losses['cmg_loss'](recons['cmg_recons'], targets['cmg_targets'])
        img_loss1 = self.losses['img_loss1'](recons['img_recons'], targets['img_targets'])
        img_loss2 = self.losses['img_loss2'](recons['img_recons'], targets['img_targets'])
        step_loss = cmg_loss + self.img_lambda1 * img_loss1 + self.img_lambda2 * img_loss2
        step_metrics = {'cmg_loss': cmg_loss, 'img_loss1': img_loss1, 'img_loss2': img_loss2}
        return recons, step_loss, step_metrics

    @staticmethod
    def _get_slice_metrics(img_recons, img_targets):

        img_recons = img_recons.detach()  # Just in case.
        img_targets = img_targets.detach()

        max_range = img_targets.max() - img_targets.min()
        slice_ssim = ssim_loss(img_recons, img_targets, max_val=max_range)
        slice_psnr = psnr_loss(img_recons, img_targets, data_range=max_range)
        slice_nmse = nmse_loss(img_recons, img_targets)

        return {'slice_ssim': slice_ssim, 'slice_nmse': slice_nmse, 'slice_psnr': slice_psnr}

    @staticmethod
    def _get_accel_slice_metrics(img_recons, img_targets, acceleration):

        img_recons = img_recons.detach()  # Just in case.
        img_targets = img_targets.detach()

        max_range = img_targets.max() - img_targets.min()
        slice_ssim = ssim_loss(img_recons, img_targets, max_val=max_range)
        slice_psnr = psnr_loss(img_recons, img_targets, data_range=max_range)
        slice_nmse = nmse_loss(img_recons, img_targets)

        if acceleration == 2:
            slice_ssim_2 = slice_ssim
            slice_psnr_2 = slice_psnr
            slice_nmse_2 = slice_nmse
            out_dict = {'slice_ssim_2': slice_ssim_2, 'slice_nmse_2': slice_nmse_2, 'slice_psnr_2': slice_psnr_2}
        elif acceleration == 4:
            slice_ssim_4 = slice_ssim
            slice_psnr_4 = slice_psnr
            slice_nmse_4 = slice_nmse
            out_dict = {'slice_ssim_4': slice_ssim_4, 'slice_nmse_4': slice_nmse_4, 'slice_psnr_4': slice_psnr_4}
        elif acceleration == 8:
            slice_ssim_8 = slice_ssim
            slice_psnr_8 = slice_psnr
            slice_nmse_8 = slice_nmse
            out_dict = {'slice_ssim_8': slice_ssim_8, 'slice_nmse_8': slice_nmse_8, 'slice_psnr_8': slice_psnr_8}

        return out_dict


    def _get_epoch_outputs(self, epoch, epoch_loss, epoch_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        epoch_loss = torch.stack(epoch_loss)
        is_finite = torch.isfinite(epoch_loss)
        num_nans = (is_finite.size(0) - is_finite.sum()).item()

        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')
            epoch_loss = torch.mean(epoch_loss[is_finite]).item()
        else:
            epoch_loss = torch.mean(epoch_loss).item()

        for key, value in epoch_metrics.items():
            epoch_metric = torch.stack(value)
            is_finite = torch.isfinite(epoch_metric)
            num_nans = (is_finite.size(0) - is_finite.sum()).item()

            if num_nans > 0:
                self.logger.warning(f'Epoch {epoch} {mode} {key}: {num_nans} NaN values present in {num_slices} slices')
                epoch_metrics[key] = torch.mean(epoch_metric[is_finite]).item()
            else:
                epoch_metrics[key] = torch.mean(epoch_metric).item()

        return epoch_loss, epoch_metrics

    def _log_step_outputs(self, epoch, step, step_loss, step_metrics, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_loss.item():.4e}')
        for key, value in step_metrics.items():
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d}: {mode} {key}: {value.item():.4e}')

    def _log_epoch_outputs(self, epoch, epoch_loss, epoch_metrics, elapsed_secs, training=True, verbose = True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} {mode}. loss: {epoch_loss:.4e}, '
                         f'Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode}_epoch_loss', scalar_value=epoch_loss, global_step=epoch)

        if verbose:
            for key, value in epoch_metrics.items():
                self.logger.info(f'Epoch {epoch:03d} {mode}. {key}: {value:.4e}')
                self.writer.add_scalar(f'{mode}_epoch_{key}', scalar_value=value, global_step=epoch)

    def _plot_trainval(self, epoch, train_epoch_loss, val_epoch_loss):
        self.writer.add_scalars('Loss', {'Train loss': train_epoch_loss,
                                         'Val loss': val_epoch_loss}, global_step=epoch)