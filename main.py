from utils.run_utils import create_arg_parser
from train.training import train_model
from train.trainer import Trainer

# Please try to use logging better. Current logging is rather badly managed.


if __name__ == '__main__':
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.01,  # Mostly for debugging purposes. Ratio of datasets to use.
        num_workers=0,  # Use 1 or 2 when training for the full dataset. Use 0 for sending data to GPU in data loader.
        init_lr=1E-4,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=0,  # Set to None for CPU mode.
        num_epochs=40,
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
        amp_fac=10000.0
    )

    parser = create_arg_parser(**defaults).parse_args()

    trainer = Trainer(parser)
    trainer.train_model()
