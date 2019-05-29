import torch

from pathlib import Path

from utils.run_utils import create_arg_parser
from train.training import train_model
from utils.run_utils import initialize, save_dict_as_json, get_logger

# Please try to use logging better. Current logging is rather badly managed.


# Allow multiprocessing on DataLoader.
# For some reason, multiprocessing causes problems with which GPU is initialized...
# Also when multiple GPUs are present. Still figuring out why.
# Maybe using a class will solve the problem...

# Try out SSIM and MS-SSIM as loss functions. They appear to be effective in getting fine-grained features,
# unlike L1.


def main():
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.1,  # Mostly for debugging purposes.
        num_workers=1,
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
        max_imgs=1,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        converted=True,
        # amp_fac=1E4,  # Amplification factor to prevent numerical underflow.
        pin_memory=False
    )

    # Replace with a proper argument parsing function later.
    args = create_arg_parser(**defaults).parse_args()

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Saving peripheral variables in args to reduce clutter.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device


if __name__ == '__main__':
    main()
