import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.data_loaders import create_prefetch_data_loaders

from train.subsample import RandomMaskFunc, UniformMaskFunc
from data.xnet_inputs import PreProcessXNet
from data.xnet_outputs import PostProcessXNet

from train.new_model_trainers.xnet_trainer import XNetModelTrainer
from models.new_xnet_model import XNet
from metrics.new_1d_ssim import SSIMLoss, LogSSIMLoss
from metrics.x_loss import AlignmentLoss


def train_xnet(args):
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

    arguments = vars(args)  # Placed here for backward compatibility and convenience.
    args.center_fractions_train = arguments.get('center_fractions_train', arguments.get('center_fractions'))
    args.center_fractions_val = arguments.get('center_fractions_val', arguments.get('center_fractions'))
    args.accelerations_train = arguments.get('accelerations_train', arguments.get('accelerations'))
    args.accelerations_val = arguments.get('accelerations_val', arguments.get('accelerations'))

    if args.random_sampling:
        train_mask_func = RandomMaskFunc(args.center_fractions_train, args.accelerations_train)
        val_mask_func = RandomMaskFunc(args.center_fractions_val, args.accelerations_val)
    else:
        train_mask_func = UniformMaskFunc(args.center_fractions_train, args.accelerations_train)
        val_mask_func = UniformMaskFunc(args.center_fractions_val, args.accelerations_val)

    input_train_transform = PreProcessXNet(mask_func=train_mask_func, challenge=args.challenge, device=device,
                                           augment_data=args.augment_data, use_seed=False, crop_center=args.crop_center)
    input_val_transform = PreProcessXNet(mask_func=val_mask_func, challenge=args.challenge, device=device,
                                         augment_data=False, use_seed=True, crop_center=args.crop_center)

    output_train_transform = PostProcessXNet(challenge=args.challenge)
    output_val_transform = PostProcessXNet(challenge=args.challenge)

    # DataLoaders
    train_loader, val_loader = create_prefetch_data_loaders(args)

    losses = dict(
        phase_loss=nn.MSELoss(),
        # img_loss=SSIMLoss(filter_size=7).to(device=device)
        img_loss=LogSSIMLoss(filter_size=5).to(device=device),
        # img_loss=nn.L1Loss()
        x_loss=AlignmentLoss()
    )

    data_chans = 1 if args.challenge == 'singlecoil' else 15  # Multicoil has 15 coils with 2 for real/imag

    model = XNet(in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
                 num_depth_blocks=args.num_depth_blocks, dilation=args.dilation, res_scale=args.res_scale).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_red_epochs, gamma=args.lr_red_rate)

    trainer = XNetModelTrainer(args, model, optimizer, train_loader, val_loader, input_train_transform,
                               input_val_transform, output_train_transform, output_val_transform, losses, scheduler)

    try:
        trainer.train_model()
    except KeyboardInterrupt:
        trainer.writer.close()
        logger.warning(f'Closing TensorBoard writer and flushing remaining outputs due to KeyboardInterrupt.')


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
        center_fractions_train=[0.08, 0.04],
        accelerations_train=[4, 8],
        center_fractions_val=[0.08, 0.04],
        accelerations_val=[4, 8],
        random_sampling=True,
        verbose=False,
        use_gt=True,
        augment_data=False,
        crop_center=True,

        # Model specific parameters.
        train_method='X',  # Weighted semi-k-space to complex-valued image.
        num_pool_layers=4,
        num_depth_blocks=2,
        dilation=1,
        res_scale=0.1,

        chans=64,
        img_lambda=0,
        phase_lambda=0,

        # TensorBoard related parameters.
        max_images=8,  # Maximum number of images to save.
        shrink_scale=1,  # Scale to shrink output image size.

        # # Channel Attention.
        # use_ca=True,
        # reduction=8,
        # use_gap=True,
        # use_gmp=False,

        # Learning rate scheduling.
        lr_red_epochs=[25, 40],
        lr_red_rate=0.25,

        # Variables that change frequently.
        use_slice_metrics=True,
        num_epochs=50,

        gpu=0,  # Set to None for CPU mode.
        num_workers=4,
        init_lr=1E-4,
        max_to_keep=1,
        # prev_model_ckpt='',

        sample_rate_train=1,
        start_slice_train=0,
        sample_rate_val=1,
        start_slice_val=0,
    )
    options = create_arg_parser(**settings).parse_args()
    train_xnet(options)
