import torch
from torch import nn, multiprocessing
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import h5py

from pathlib import Path
from collections import defaultdict

from utils.train_utils import load_model_from_checkpoint
from utils.run_utils import create_arg_parser
from data.mri_data import CustomSliceData


class ModelEvaluator:
    def __init__(self, modelI, modelK, checkpoint_path_I, checkpoint_path_K, challenge, data_loader,
                 pre_processing, mid_processing, post_processing, data_root, out_dir, device):

        assert isinstance(modelI, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(modelK, nn.Module), '`model` must be a Pytorch Module.'
        assert isinstance(data_loader, DataLoader), '`data_loader` must be a Pytorch DataLoader.'
        assert callable(pre_processing), '`pre_processing` must be a callable function.'
        assert callable(post_processing), 'post_processing must be a callable function.'
        assert challenge in ('singlecoil', 'multicoil'), 'Invalid challenge.'

        torch.autograd.set_grad_enabled(False)
        self.modelI = load_model_from_checkpoint(modelI, checkpoint_path_I).to(device)
        print(f'Loaded modelI parameters from {checkpoint_path_I}')
        self.modelI.eval()
        self.modelK = load_model_from_checkpoint(modelK, checkpoint_path_K).to(device)
        print(f'Loaded modelI parameters from {checkpoint_path_K}')
        self.modelK.eval()

        self.data_loader = data_loader
        self.challenge = challenge
        self.pre_processing = pre_processing
        self.mid_processing = mid_processing
        self.post_processing = post_processing
        self.data_root = data_root
        self.out_dir = Path(out_dir)
        self.device = device

    def _create_reconstructions(self):
        reconstructions = defaultdict(list)

        for data in tqdm(self.data_loader, total=len(self.data_loader.dataset)):

            kspace_target, target, attrs, file_name, slice_num = data
            inputs, targets, extra_params = self.pre_processing(kspace_target, target, attrs, file_name, slice_num)
            I_outputs = self.modelI(inputs)  # Use inputs.to(device) if necessary for different transforms.
            mid_outputs = self.mid_processing(I_outputs, target, extra_params)
            K_outputs = self.modelK(mid_outputs['kspace_recons'])
            recons = self.post_processing(K_outputs, extra_params)

            assert recons.dim() == 2, 'Unexpected dimensions. Batch size is expected to be 1.'

            recons = recons.cpu().numpy()
            file_name = file_name[0]
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


def main(args):
    from models.fc_unet import Unet  # Moving import line here to reduce confusion.
    from data.input_transforms import TestPreProcessCCInfo, Prefetch2Device
    from data.test_data_transforms import TestPreProcessCC
    from data.output_transforms import OutputTransformIKTest, MidTransformK
    from train.subsample import RandomMaskFunc

    # Selecting device
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f'Device {device} has been selected.')

    modelI = Unet(in_chans=args.data_chans, out_chans=args.data_chans, chans=args.I_chans,
                  num_pool_layers=args.num_pool_layers).to(device)
    modelK = Unet(in_chans=args.data_chans, out_chans=args.data_chans, chans=args.K_chans,
                  num_pool_layers=args.num_pool_layers).to(device)

    dataset = CustomSliceData(root=args.data_root, transform=Prefetch2Device(device), challenge=args.challenge,
                              sample_rate=1, start_slice=0, use_gt=True)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    mask_func = RandomMaskFunc(args.center_fractions, args.accelerations)
    divisor = 2 ** args.num_pool_layers  # For UNet size fitting.

    # This is for the validation set, not the test set. The test set requires a different pre-processing function.
    if Path(args.data_root).name.endswith('val'):
        pre_processing = TestPreProcessCCInfo(mask_func, args.challenge, args.gpu,
                                              use_seed=False, divisor=divisor)
    else:
        pre_processing = TestPreProcessCC(challenge='multicoil', device=0)

    mid_processing = MidTransformK()

    post_processing = OutputTransformIKTest()

    evaluator = ModelEvaluator(modelI, modelK, args.checkpoint_path_I, args.checkpoint_path_K, args.challenge, data_loader,
                               pre_processing, mid_processing, post_processing, args.data_root, args.out_dir, device)

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
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],

        # Model specific parameters.
        data_chans=30,
        I_chans=64,
        K_chans=32,
        num_pool_layers=5,

        # Parameters for reconstruction.
        data_root='/media/harry/fastmri/fastMRI_data/multicoil_val',
        checkpoint_path_I='/home/harry/PycharmProjects/fastMRI-kspace/checkpoints/'
                          'IMG/[IMG]GRU_P2/ckpt_G027.tar',
        checkpoint_path_K='/home/harry/PycharmProjects/fastMRI-kspace/checkpoints/'
                          'IMG/[IMG]IK/ckpt_K067.tar',

        out_dir='./IK_val'  # Change this every time! Attempted overrides will throw errors by design.
    )

    parser = create_arg_parser(**defaults).parse_args()
    main(parser)