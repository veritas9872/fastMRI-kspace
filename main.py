from utils.run_utils import create_arg_parser
from train.training import train_model
from train.trainer import Trainer
from eda.pp_unet_model import UnetModel
from data.post_processing import TrainBatchTransform
# Please try to use logging better. Current logging is rather badly managed.

# A hack to go around bug with multi-processing with multiple GPUs.
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # This is the true device ID. "0, 1" for multiple
# cuda.set_device(0)

# Allow multiprocessing on DataLoader.
# For some reason, multiprocessing causes problems with which GPU is initialized...
# Also when multiple GPUs are present. Still figuring out why.

if __name__ == '__main__':

    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.01,  # Mostly for debugging purposes. Ratio of datasets to use.
        num_workers=1,  # Use 1 or 2 when training for the full dataset. Use 0 for sending data to GPU in data loader.
        init_lr=1E-4,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.  # Not true GPU for now.
        num_epochs=5,
        max_to_keep=1,
        verbose=False,
        save_best_only=True,
        data_root='/media/user/Data2/compFastMRI',  # Using compressed dataset for better I/O performance.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        # I don't know why having more than 1 image doesn't work but it causes problems.
        max_imgs=1,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        converted=True,
        amp_fac=1E8,  # Amplification factor to prevent numerical underflow.
    )

    parser = create_arg_parser(**defaults).parse_args()

    model = UnetModel(in_chans=30, out_chans=30, chans=parser.chans, num_pool_layers=parser.num_pool_layers,
                      post_processing=TrainBatchTransform())

    trainer = Trainer(parser, model=model)
    trainer.train_model()
