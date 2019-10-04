import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.data_loaders import create_prefetch_data_loaders, create_prefetch_data_loaders_end

from train.subsample import RandomMaskFunc, UniformMaskFunc
from data.rss_inputs import PreProcessRSS
from data.rss_outputs import PostProcessRSS

from train.new_model_trainers.img_to_rss import ModelTrainerRSS
from metrics.new_1d_ssim import SSIMLoss, LogSSIMLoss
from metrics.combination_losses import L1SSIMLoss
from models.hj1_net import UNet


def train_img_to_rss(args):
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

    input_train_transform = PreProcessRSS(mask_func=train_mask_func, challenge=args.challenge, device=device,
                                          augment_data=args.augment_data, use_seed=False)
    input_val_transform = PreProcessRSS(mask_func=val_mask_func, challenge=args.challenge, device=device,
                                        augment_data=False, use_seed=True)

    output_train_transform = PostProcessRSS(challenge=args.challenge, residual_rss=args.residual_rss)
    output_val_transform = PostProcessRSS(challenge=args.challenge, residual_rss=args.residual_rss)

    # DataLoaders
    # train_loader, val_loader = create_prefetch_data_loaders(args)
    train_loader, val_loader = create_prefetch_data_loaders_end(args)

    losses = dict(
        # rss_loss=SSIMLoss(filter_size=7).to(device=device)
        # rss_loss=SSIMLoss(filter_size=7, max_val=0.0002).to(device=device)
        rss_loss=nn.L1Loss()
        # rss_loss=L1SSIMLoss(filter_size=7, l1_ratio=args.l1_ratio).to(device=device)
    )

    # model = UNet(in_chans=15, out_chans=1, chans=args.chans, num_pool_layers=args.num_pool_layers,
    #              num_depth_blocks=args.num_depth_blocks, res_scale=args.res_scale, use_residual=args.use_residual,
    #              use_ca=args.use_ca, reduction=args.reduction, use_gap=args.use_gap, use_gmp=args.use_gmp).to(device)

    model = UNet(in_chans=15, out_chans=1, chans=args.chans, num_pool_layers=args.num_pool_layers,
                 num_depth_blocks=args.num_depth_blocks, res_scale=args.res_scale, use_residual=args.use_residual,
                 use_ca=args.use_ca, reduction=args.reduction, use_gap=args.use_gap, use_gmp=args.use_gmp,
                 use_sa=False, sa_kernel_size=7, sa_dilation=1,
                 use_cap=False, use_cmp=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_red_epochs, gamma=args.lr_red_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=args.lr_step_rate)

    scheduler = None

    trainer = ModelTrainerRSS(args, model, optimizer, train_loader, val_loader, input_train_transform,
                              input_val_transform, output_train_transform, output_val_transform, losses, scheduler)

    try:
        trainer.train_model()
        # trainer.val_model()
    except KeyboardInterrupt:
        trainer.writer.close()
        logger.warning('Closing summary writer due to KeyboardInterrupt.')


if __name__ == '__main__':
    project_name = 'fastMRI-kspace'
    assert Path.cwd().name == project_name, f'Current working directory set at {Path.cwd()}, not {project_name}!'

    settings = dict(
        # Variables that almost never change.
        challenge='multicoil',
        data_root='/media/user/Data/compFastMRI',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,  # This MUST be 1 for now.
        save_best_only=False,
        smoothing_factor=8,

        # Variables that occasionally change.
        center_fractions_train=[0.08, 0.04],
        accelerations_train=[4, 8],
        # When using single acceleration for train and two accelerations for validation,
        # please remember that the validation loss is calculated for both accelerations,
        # including the one that the model was not trained for.
        # This may result in the checkpoint not being saved,
        # even though performance on one acceleration improves significantly.
        center_fractions_val=[0.08, 0.04],
        accelerations_val=[4, 8],

        random_sampling=True,
        num_pool_layers=2,
        verbose=False,
        use_gt=True,

        # Model specific parameters.
        train_method='I2R',
        chans=128,
        use_residual=False,
        residual_rss=False,
        # l1_ratio=0.5,
        num_depth_blocks=32,
        res_scale=0.1,
        augment_data=True,
        crop_center=True,

        # TensorBoard related parameters.
        max_images=10,  # Maximum number of images to save.
        shrink_scale=1,  # Scale to shrink output image size.

        # Channel Attention.
        use_ca=True,
        reduction=16,
        use_gap=True,
        use_gmp=False,

        # Learning rate scheduling.
        lr_red_epochs=[40, 55],
        lr_red_rate=0.25,

        lr_step_epochs=1,
        lr_step_rate=0.75,

        # Variables that change frequently.
        use_slice_metrics=True,
        num_epochs=100,

        gpu=0,  # Set to None for CPU mode.
        num_workers=2,
        init_lr=1E-4,
        max_to_keep=10,
        prev_model_ckpt='checkpoints/I2R/Trial 61  2019-09-19 00-51-19/ckpt_055.tar',

        sample_rate_train=1,
        start_slice_train=2,
        end_slice_train=5,
        sample_rate_val=1,
        start_slice_val=2,
        end_slice_val=5,

    )
    options = create_arg_parser(**settings).parse_args()
    train_img_to_rss(options)