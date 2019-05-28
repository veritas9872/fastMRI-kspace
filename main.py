from utils.run_utils import create_arg_parser
from train.training import train_model

# Please try to use logging better. Current logging is rather badly managed.


if __name__ == '__main__':
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=1,  # Mostly for debugging purposes. Ratio of datasets to use.
        num_workers=1,  # Use 1 or 2 when training for the full dataset. Use 0 for sending data to GPU in data loader.
        init_lr=1E-3,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=20,
        max_to_keep=1,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/F/compFastMRI',  # Using compressed dataset for better I/O performance.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        # I don't know why having more than 1 image doesn't work but it causes problems.
        max_imgs=2,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        converted=True,
        amp_fac=1E8,  # Amplification factor to prevent numerical underflow.
        previous_model='checkpoints/Trial 03  2019-05-16 18-20-55/ckpt_012.tar'
    )

    parser = create_arg_parser(**defaults).parse_args()

    train_model(parser)
