from utils.run_utils import create_arg_parser
from train.training import train_model


if __name__ == '__main__':
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.1,  # Mostly for debugging purposes.
        num_workers=1,  # Use 1 or 2 when training for the full dataset.
        init_lr=1E-3,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=10,
        max_to_keep=2,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/F/compFastMRI',  # Using compressed dataset for better I/O performance.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_imgs=4,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        converted=True
    )

    parser = create_arg_parser(**defaults).parse_args()

    train_model(parser)
