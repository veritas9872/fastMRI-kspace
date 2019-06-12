import torch
import torch.optim as optim
import torch.nn as nn

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from data.pre_processing import KInputTransform, KInputSliceTransform
from utils.train_utils import create_data_loaders
from train.subsample import MaskFunc
from train.processing import SingleBatchOutputTransform, OutputBatchMaskTransform
from train.mask_model_trainer import ModelTrainerK
from train.metrics import CustomL1Loss, CustomL2Loss
from train.loss import CustomLoss
from eda.unet_model import ResidualUnetModel


# Try out SSIM and MS-SSIM as loss functions. They appear to be effective in getting fine-grained features,
# unlike L1.


def main():
    defaults = dict(
        batch_size=2,
        sample_rate=1,  # Mostly for debugging purposes.
        num_workers=1,
        init_lr=1E-4,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=0,  # Set to None for CPU mode.
        num_epochs=50,
        max_to_keep=2,
        verbose=False,
        save_best_only=True,
        data_root='/media/user/Data2/compFastMRI',  # Using compressed dataset for better I/O performance.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_images=4,  # Maximum number of images to save.
        chans=64,
        num_pool_layers=4,
        converted=True,
        pin_memory=False,
        add_graph=False
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

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    # Input transforms. These are on a slice basis.
    # UNET architecture requires that all inputs be dividable by some power of 2.
    divisor = 2 ** args.num_pool_layers

    mask_func = MaskFunc(args.center_fractions, args.accelerations)

    input_slice_train_transform = KInputTransform(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=False, divisor=divisor)

    input_slice_val_transform = KInputTransform(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(
        args=args, train_transform=input_slice_train_transform, val_transform=input_slice_val_transform)

    # Loss Function and output post-processing functions.
    if args.batch_size == 1:
        loss_func = nn.L1Loss(reduction='mean')
        output_batch_transform = SingleBatchOutputTransform()
    elif args.batch_size > 1:
        loss_func = CustomL2Loss(reduction='mean')
        output_batch_transform = OutputBatchMaskTransform()
    else:
        raise RuntimeError('Invalid batch size.')

    # Define model.
    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
    model = ResidualUnetModel(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
                              num_pool_layers=args.num_pool_layers).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    trainer = ModelTrainerK(args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                            post_processing=output_batch_transform, loss_func=loss_func)

    trainer.train_model()


if __name__ == '__main__':
    main()
