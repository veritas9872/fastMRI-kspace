import torch
from torch import nn
from torch.utils.data import DataLoader

from models.fc_unet import FCUnet, Unet
import numpy as np
from tqdm import tqdm
import h5py
from utils.train_utils import load_model_from_checkpoint
from utils.run_utils import create_arg_parser

from data.test_data_transforms import TestPrefetch2Device, TestPreProcessCC, TestOutputTransformCC
from data.mri_data import CustomSliceTestData

from pathlib import Path
from collections import defaultdict


class ModelEvaluator:
    def __init__(self, model, checkpoint_path, pre_processing, post_processing, data_dir, out_dir, device):

        assert isinstance(model, nn.Module)
        assert callable(pre_processing), '`pre_processing` must be a callable function.'
        assert callable(post_processing), 'post_processing must be a callable function.'

        torch.autograd.set_grad_enabled(False)
        self.model = load_model_from_checkpoint(model, checkpoint_path).to(device)
        print(f'Loaded model parameters from {checkpoint_path}')
        self.model.eval()

        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.data_dir = data_dir
        self.out_dir = Path(out_dir)
        self.device = device

    def _create_reconstructions(self, data_loader):
        reconstructions = defaultdict(list)
        for ds_slices, attrs, file_names, s_idxs in tqdm(data_loader):
            inputs, extra_params = self.pre_processing(ds_slices.to(device=self.device), file_names, s_idxs)
            recons = self.model(inputs)
            recons = self.post_processing(recons, extra_params)['cmg_recons'].cpu().numpy()

            for idx in range(recons.shape[0]):
                file_name = Path(file_names[idx]).name
                reconstructions[file_name].append((int(s_idxs[idx]), recons[idx, ...].squeeze()))

        import ipdb; ipdb.set_trace()
        reconstructions = {
            file_name: np.stack([recon for _, recon in sorted(recons_list)])
            for file_name, recons_list in reconstructions.items()
        }

        return reconstructions

    def _create_data_loader(self):
        dataset = CustomSliceTestData(root=self.data_dir, transform=None, challenge='multicoil')
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
        return data_loader

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
        data_loader = self._create_data_loader()
        print('Beginning Reconstruction.')
        reconstructions = self._create_reconstructions(data_loader)
        print('Beginning Saving.')
        self._save_reconstructions(reconstructions)


def main(model, args):
    # Selecting device
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f'Device {device} has been selected.')

    pre_processing = TestPreProcessCC(challenge='multicoil', device=0)
    post_processing = TestOutputTransformCC()

    evaluator = ModelEvaluator(
        model, args.checkpoint_path, pre_processing, post_processing, args.data_dir, args.out_dir, device)

    evaluator.create_and_save_reconstructions()


if __name__ == '__main__':
    print(f'Current Working Directory: {Path.cwd()}')
    defaults = dict(
        gpu=0,  # Set to None for CPU mode.
        data_dir='/media/harry/fastmri/fastMRI_data/multicoil_test_test',
        checkpoint_path='/home/harry/PycharmProjects/fastMRI-kspace/checkpoints/'
                        'IMG/[IMG]ResUnet/ckpt_052.tar',

        out_dir='./ResUnet'  # Change this every time! Attempted overrides will throw errors by design.
    )

    # Model settings
    settings = dict(
        data_chans=30,  # multi_coil,
        chans=64,  # Unet channel size
        num_pool_layers=5,  # Unet depth
    )

    parser = create_arg_parser(**defaults).parse_args()

    # Change this part when a different model is being used.
    model = Unet(in_chans=settings['data_chans'], out_chans=settings['data_chans'], chans=settings['chans'],
                 num_pool_layers=settings['num_pool_layers'])

    main(model, parser)
