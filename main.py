from utils.run_utils import create_arg_parser
from train.training import train_model


if __name__ == '__main__':
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.1,
        num_workers=1,
        init_lr=1E-3,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=2,
        max_to_keep=1,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/E/fastMRI',
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_imgs=0,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4
    )

    parser = create_arg_parser(**defaults).parse_args()

    train_model(parser)
