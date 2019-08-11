import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.data_loaders import create_prefetch_data_loaders

from train.subsample import RandomMaskFunc, UniformMaskFunc
from data.input_transforms import PreProcessWSK
from data.output_transforms import PostProcessWSemiK

from train.new_model_trainers.img_only import ModelTrainerIMG
from models.unet_no_norm import UNet
from metrics.new_1d_ssim import SSIMLoss, LogSSIMLoss


def no_weight(inputs):
    return 1


def train_img_only(args):
    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_root)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / args.train_method
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_root)
    log_path.mkdir(exist_ok=True)

    log_path = log_path / args.train_method
    log_path.mkdir(exist_ok=True)

    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Input transforms. These are on a per-slice basis.
    # UNET architecture requires that all inputs be dividable by some power of 2.
    divisor = 2 ** args.num_pool_layers

    if args.random_sampling:
        mask_func = RandomMaskFunc(args.center_fractions, args.accelerations)
    else:
        mask_func = UniformMaskFunc(args.center_fractions, args.accelerations)

    input_train_transform = PreProcessWSK(mask_func=mask_func, weight_func=no_weight, challenge=args.challenge,
                                          device=device, use_seed=False, divisor=divisor)
    input_val_transform = PreProcessWSK(mask_func=mask_func, weight_func=no_weight, challenge=args.challenge,
                                        device=device, use_seed=True, divisor=divisor)

    output_train_transform = PostProcessWSemiK(weighted=False, replace=args.replace, residual_acs=args.residual_acs)
    output_val_transform = PostProcessWSemiK(weighted=False, replace=args.replace, residual_acs=args.residual_acs)

    # DataLoaders
    train_loader, val_loader = create_prefetch_data_loaders(args)

    losses = dict(
        # img_loss=LogSSIMLoss(filter_size=7).to(device)
        # img_loss=SSIMLoss(filter_size=7).to(device=device)
        img_loss=nn.L1Loss()
    )

    model = UNet(
        in_chans=30, out_chans=30, chans=args.chans, num_pool_layers=args.num_pool_layers,
        num_depth_blocks=args.num_depth_blocks, interp_mode=args.interp_mode, use_residual=args.use_residual,
        use_ca=args.use_ca, reduction=args.reduction, use_gap=args.use_gap, use_gmp=args.use_gmp).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_red_epochs, gamma=args.lr_red_rate)

    trainer = ModelTrainerIMG(args, model, optimizer, train_loader, val_loader, input_train_transform,
                              input_val_transform, output_train_transform, output_val_transform, losses, scheduler)

    try:
        trainer.train_model()
    except KeyboardInterrupt:
        trainer.writer.close()
        logger.warning('Closing summary writer due to KeyboardInterrupt.')


if __name__ == '__main__':
    project_name = 'fastMRI-kspace'
    assert Path.cwd().name == project_name, f'Current working directory set at {Path.cwd()}, not {project_name}!'

    settings = dict(
        # Variables that almost never change.
        challenge='multicoil',
        data_root='/media/veritas/D/FastMRI',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,  # This MUST be 1 for now.
        save_best_only=True,
        smoothing_factor=8,

        # Variables that occasionally change.
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        random_sampling=True,
        num_pool_layers=5,
        verbose=False,
        use_gt=True,

        # Model specific parameters.
        train_method='S2I',  # Semi-k-space to complex-valued image.
        # num_groups=16,  # Maybe try 16 now since chans is 64.
        chans=64,
        # negative_slope=0.1,
        interp_mode='nearest',
        replace=False,
        use_residual=False,
        residual_acs=True,
        num_depth_blocks=1,

        # TensorBoard related parameters.
        max_images=8,  # Maximum number of images to save.
        shrink_scale=1,  # Scale to shrink output image size.

        # Channel Attention.
        use_ca=False,
        reduction=8,
        use_gap=False,
        use_gmp=False,

        # Learning rate scheduling.
        lr_red_epochs=[20, 25],
        lr_red_rate=0.1,

        # Variables that change frequently.
        use_slice_metrics=True,
        num_epochs=30,

        gpu=1,  # Set to None for CPU mode.
        num_workers=4,
        init_lr=2E-2,
        max_to_keep=1,
        # prev_model_ckpt='',

        sample_rate_train=1,
        start_slice_train=0,
        sample_rate_val=1,
        start_slice_val=0,
    )
    arguments = create_arg_parser(**settings).parse_args()
    train_img_only(arguments)
