import torch
from torch import nn, multiprocessing
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import h5py

from pathlib import Path
from collections import defaultdict

from utils.train_utils import load_model_from_checkpoint
from utils.data_loaders import temp_collate_fn
from utils.run_utils import create_arg_parser
from data.mri_data import CustomSliceData


class ModelEvaluator:
    def __init__(self, model, checkpoint_path, challenge, data_loader,
                 pre_processing, post_processing, data_root, out_dir, device):

        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(data_loader, DataLoader), '`data_loader` must be a Pytorch DataLoader.'
        assert callable(pre_processing), '`pre_processing` must be a callable function.'
        assert callable(post_processing), 'post_processing must be a callable function.'
        assert challenge in ('singlecoil', 'multicoil'), 'Invalid challenge.'
        assert not Path(out_dir).exists(), 'Output directory already exists!'

        torch.autograd.set_grad_enabled(False)
        self.model = load_model_from_checkpoint(model, checkpoint_path).to(device)
        print(f'Loaded model parameters from {checkpoint_path}')
        self.model.eval()

        self.data_loader = data_loader
        self.challenge = challenge
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.data_root = data_root
        self.out_dir = Path(out_dir)
        self.device = device

    def _create_reconstructions(self):
        reconstructions = defaultdict(list)

        for data in tqdm(self.data_loader, total=len(self.data_loader.dataset)):
            kspace_target, target, attrs, file_name, slice_num = data
            inputs, targets, extra_params = self.pre_processing(kspace_target, target, attrs, file_name, slice_num)
            outputs = self.model(inputs)  # Use inputs.to(device) if necessary for different transforms.
            recons = self.post_processing(outputs, targets, extra_params)
            assert recons.dim() == 2, 'Unexpected dimensions. Batch size is expected to be 1.'

            recons = recons.cpu().numpy()
            file_name = Path(file_name).name
            reconstructions[file_name].append((int(slice_num), recons))

        reconstructions = {
            file_name: np.stack([recon for _, recon in sorted(recons_list)])
            for file_name, recons_list in reconstructions.items()
        }

        return reconstructions

    def _save_reconstructions(self, reconstructions):
        """
        Saves the reconstructions from a model into h5 files that is appropriate for submission
        to the leaderboard. Also includes compression for faster data transfer.
        Args:
            reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
                corresponding reconstructions (of shape num_slices x height x width).
        """
        self.out_dir.mkdir(exist_ok=False)  # Throw an error to prevent overwriting.
        gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
        for file_name, recons in tqdm(reconstructions.items()):
            with h5py.File(self.out_dir / file_name, mode='x', libver='latest') as f:  # Overwriting throws an error.
                f.create_dataset('reconstruction', data=recons, **gzip)

    def create_and_save_reconstructions(self):
        print('Beginning Reconstruction.')
        reconstructions = self._create_reconstructions()
        print('Beginning Saving.')
        self._save_reconstructions(reconstructions)


class MultiAccelerationModelEvaluator:
    """
    Model evaluator for evaluating models trained on different accelerations.
    Does not separate cases for weight suppression, only for 4-fold and 8-fold accelerations.
    """
    def __init__(self, model, checkpoint_path_4, checkpoint_path_8, challenge, data_loader,
                 pre_processing, post_processing, data_root, out_dir, device):

        assert isinstance(model, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(data_loader, DataLoader), '`data_loader` must be a Pytorch DataLoader.'
        assert callable(pre_processing), '`pre_processing` must be a callable function.'
        assert callable(post_processing), 'post_processing must be a callable function.'
        assert challenge in ('singlecoil', 'multicoil'), 'Invalid challenge.'
        assert not Path(out_dir).exists(), 'Output directory already exists!'

        torch.autograd.set_grad_enabled(False)

        self.model = model.eval()
        self.checkpoint_path_4 = checkpoint_path_4
        self.checkpoint_path_8 = checkpoint_path_8
        self.data_loader = data_loader
        self.challenge = challenge
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.data_root = data_root
        self.out_dir = Path(out_dir)
        self.device = device

    def _create_partial_reconstructions(self, checkpoint_path, acceleration):
        reconstructions = defaultdict(list)
        self.model = load_model_from_checkpoint(self.model, checkpoint_path).to(self.device).eval()

        for data in tqdm(self.data_loader, total=len(self.data_loader.dataset)):
            kspace_target, target, attrs, file_name, slice_num = data
            inputs, targets, extra_params = self.pre_processing(kspace_target, target, attrs, file_name, slice_num)

            if extra_params['acceleration'] != acceleration:
                continue  # Not very efficient since data is still loaded and sent to GPU device. However, it will do.

            outputs = self.model(inputs)  # Use inputs.to(device) if necessary for different transforms.
            recons = self.post_processing(outputs, extra_params)
            assert recons.dim() == 2, 'Unexpected dimensions. Batch size is expected to be 1.'

            recons = recons.cpu().numpy()
            file_name = Path(file_name).name
            reconstructions[file_name].append((int(slice_num), recons))

        reconstructions = {
            file_name: np.stack([recon for _, recon in sorted(recons_list)])
            for file_name, recons_list in reconstructions.items()
        }

        return reconstructions

    def _save_reconstructions(self, reconstructions):
        """
        Saves the reconstructions from a model into h5 files that is appropriate for submission
        to the leaderboard. Also includes compression for faster data transfer.
        Args:
            reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
                corresponding reconstructions (of shape num_slices x height x width).
        """
        self.out_dir.mkdir(exist_ok=False)  # Throw an error to prevent overwriting.
        gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
        for file_name, recons in tqdm(reconstructions.items()):
            with h5py.File(self.out_dir / file_name, mode='x', libver='latest') as f:  # Overwriting throws an error.
                f.create_dataset('reconstruction', data=recons, **gzip)

    def create_and_save_reconstructions(self):
        print('Beginning Reconstruction.')
        reconstructions_4 = self._create_partial_reconstructions(checkpoint_path=self.checkpoint_path_4, acceleration=4)
        reconstructions_8 = self._create_partial_reconstructions(checkpoint_path=self.checkpoint_path_8, acceleration=8)
        print('Beginning Saving.')

        # Rather slow and inefficient but this makes for better looking code.
        reconstructions = dict(**reconstructions_4, **reconstructions_8)
        self._save_reconstructions(reconstructions)


def main(args):
    from models.edsr_unet import UNet  # Moving import line here to reduce confusion.
    from data.input_transforms import PreProcessIMG, Prefetch2Device
    from data.rss_inputs import PreProcessRSS
    from eval.input_test_transform import PreProcessTestIMG, PreProcessValIMG, PreProcessTestRSS
    from eval.output_test_transforms import PostProcessTestIMG, PostProcessTestRSS
    from train.subsample import RandomMaskFunc

    # Selecting device
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f'Device {device} has been selected.')

    # data_chans = 1 if args.challenge == 'singlecoil' else 15
    model = UNet(in_chans=15, out_chans=1, chans=args.chans, num_pool_layers=args.num_pool_layers,
                 num_depth_blocks=args.num_depth_blocks, res_scale=args.res_scale, use_residual=args.use_residual,
                 use_ca=args.use_ca, reduction=args.reduction, use_gap=args.use_gap, use_gmp=args.use_gmp).to(device)

    dataset = CustomSliceData(root=args.data_root, transform=Prefetch2Device(device),
                              challenge=args.challenge, sample_rate=1, start_slice=0, use_gt=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, collate_fn=temp_collate_fn, pin_memory=False)

    mask_func = RandomMaskFunc(args.center_fractions, args.accelerations)
    # divisor = 2 ** args.num_pool_layers  # For UNet size fitting.

    # This is for the validation set, not the test set. The test set requires a different pre-processing function.
    if Path(args.data_root).name.endswith('val'):
        # pre_processing = PreProcessIMG(mask_func=mask_func, challenge=args.challenge, device=device,
        #                                augment_data=False, use_seed=True, crop_center=True)
        pre_processing = PreProcessRSS(mask_func=mask_func, challenge=args.challenge, device=device,
                                       augment_data=False, use_seed=True)
    elif Path(args.data_root).name.endswith('test_v2'):
        pre_processing = PreProcessTestRSS(challenge=args.challenge, device=device)
        # pre_processing = PreProcessTestIMG(challenge=args.challenge, device=device, crop_center=True)
    else:
        raise NotImplementedError('Invalid data root. If using the original test set, please change to test_v2.')

    # post_processing = PostProcessTestIMG(challenge=args.challenge)
    post_processing = PostProcessTestRSS(challenge=args.challenge, residual_rss=args.residual_rss)

    # Single acceleration, single model version.
    evaluator = ModelEvaluator(model, args.checkpoint_path, args.challenge, data_loader,
                               pre_processing, post_processing, args.data_root, args.out_dir, device)

    # evaluator = MultiAccelerationModelEvaluator(
    #     model=model, checkpoint_path_4=args.checkpoint_path_4, checkpoint_path_8=args.checkpoint_path_8,
    #     challenge=args.challenge, data_loader=data_loader, pre_processing=pre_processing,
    #     post_processing=post_processing, data_root=args.data_root, out_dir=args.out_dir, device=device
    # )

    evaluator.create_and_save_reconstructions()


if __name__ == '__main__':
    print(f'Current Working Directory: {Path.cwd()}')
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method(method='spawn')

    defaults = dict(
        gpu=0,  # Set to None for CPU mode.
        challenge='multicoil',
        num_workers=4,

        # Parameters for validation set evaluation.
        center_fractions=[0.08],
        accelerations=[4],

        # Model specific parameters.
        chans=64,
        num_pool_layers=3,
        num_depth_blocks=32,
        res_scale=0.1,
        use_residual=False,
        residual_rss=True,
        use_ca=True,
        reduction=16,
        use_gap=True,
        use_gmp=False,

        # Parameters for reconstruction.
        data_root='/media/veritas/D/FastMRI/multicoil_val',
        checkpoint_path=
        '/home/veritas/PycharmProjects/fastMRI-kspace/checkpoints/I2R/Trial 01  2019-08-23 17-46-32/ckpt_047.tar',
        # checkpoint_path_4=
        # '/home/veritas/PycharmProjects/fastMRI-kspace/checkpoints/I2R/Trial 09  2019-08-28 11-34-33/ckpt_001.tar',
        # checkpoint_path_8='',

        # Change this every time! Attempted overrides will throw errors by design.
        out_dir='./i2r_4_check_old'
    )

    parser = create_arg_parser(**defaults).parse_args()
    main(parser)
