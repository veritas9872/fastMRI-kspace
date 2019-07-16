import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_custom_data_loaders

from train.subsample import MaskFunc, UniformMaskFunc
from data.input_transforms import Prefetch2Device, WeightedPreProcessK
from data.output_transforms import WeightedReplacePostProcess

from train.model_trainers.new_model_trainer_K2I import ModelTrainerK2I
from models.new_skip_unet import UNetSkipGN
from metrics.combination_losses import L1CSSIM7

"""
Memo: I have found that there is a great deal of variation in performance when training.
Even under the same settings, the results can be extremely different when using small numbers of samples. 
I believe that this is because of the large degree of variation in data quality in the dataset.
Therefore, demonstrating that one method works better than another requires using a large portion of the dataset.
However, this takes a lot of time...
Using small datasets for multiple runs may also prove useful.
"""


def train_k2i(args):

    # Maybe move this to args later.
    train_method = 'W2I'  # Weighted K-space to real-valued image.

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_root)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / train_method
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_root)
    log_path.mkdir(exist_ok=True)

    log_path = log_path / train_method
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
        mask_func = MaskFunc(args.center_fractions, args.accelerations)
    else:
        mask_func = UniformMaskFunc(args.center_fractions, args.accelerations)

    # This is optimized for SSD storage.
    # Sending to device should be inside the input transform for optimal performance on HDD.
    data_prefetch = Prefetch2Device(device)

    input_train_transform = WeightedPreProcessK(mask_func, args.challenge, device, use_seed=False, divisor=divisor)
    input_val_transform = WeightedPreProcessK(mask_func, args.challenge, device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_custom_data_loaders(args, transform=data_prefetch)

    losses = dict(
        img_loss=nn.L1Loss(reduction='mean')
        # img_loss=L1CSSIM7(reduction='mean', alpha=args.alpha)
    )

    output_transform = WeightedReplacePostProcess()

    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag

    model = UNetSkipGN(
        in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
        num_groups=args.num_groups, pool_type=args.pool_type, use_skip=args.use_skip, use_att=args.use_att,
        reduction=args.reduction, use_gap=args.use_gap, use_gmp=args.use_gmp).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_red_epochs, gamma=args.lr_red_rate)

    trainer = ModelTrainerK2I(args, model, optimizer, train_loader, val_loader,
                              input_train_transform, input_val_transform, output_transform, losses, scheduler)

    trainer.train_model()


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
        num_pool_layers=4,
        save_best_only=True,
        center_fractions=[0.08],
        accelerations=[4],
        smoothing_factor=8,

        # Variables that occasionally change.
        chans=64,
        max_images=12,  # Maximum number of images to save.
        num_workers=1,
        init_lr=2E-2,
        max_to_keep=1,
        start_slice=6,
        random_sampling=True,
        verbose=False,

        # Learning rate scheduling.
        lr_red_epochs=[40, 60, 80],
        lr_red_rate=0.1,

        # Model specific parameters.
        num_groups=8,
        pool_type='avg',
        use_skip=False,
        use_att=False,
        reduction=16,
        use_gap=True,
        use_gmp=False,
        alpha=0.5,

        # Variables that change frequently.
        sample_rate=0.25,  # Ratio of the dataset to sample and use.
        num_epochs=100,
        gpu=1,  # Set to None for CPU mode.
        use_slice_metrics=True,  # This can significantly increase training time.
        # prev_model_ckpt='',
    )
    options = create_arg_parser(**settings).parse_args()
    train_k2i(options)
