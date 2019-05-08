from utils.run_utils import create_arg_parser
from train.training import train_model

"""
Notes about the number of workers to use.
The data loader will load num_workers mini-batches at once, using multiple processes.
This means that there will be multiple mini-batches being prepared at one time.
However, I have found that overwhelming the data I/O actually decreases speed when the files are being loaded.
Although this problem disappears for small datasets which can be cached in memory, 
assuming that the compressed version of the dataset is being used,
none of the full datasets can possibly be cached.
Therefore, do not fully saturate data I/O when deciding how many workers to use. 
"""


if __name__ == '__main__':
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.01,  # Mostly for debugging purposes.
        num_workers=1,
        init_lr=1E-4,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=10,
        max_to_keep=4,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/F/compFastMRI',
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_imgs=2,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4
    )

    parser = create_arg_parser(**defaults).parse_args()

    train_model(parser)
