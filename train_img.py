import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_custom_data_loaders

from train.subsample import MaskFunc
from data.input_transforms import InputTransformK
from data.output_transforms import OutputReplaceTransformK


from models.ks_unet import UnetKS
from train.model_trainers.model_trainer_IMG import ModelTrainerIMG


def train_img(args):

    # Maybe move this to args later.
    train_method = 'IMG'

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

    mask_func = MaskFunc(args.center_fractions, args.accelerations)

    train_transform = InputTransformK(mask_func, args.challenge, args.device, use_seed=False, divisor=divisor)
    val_transform = InputTransformK(mask_func, args.challenge, args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_custom_data_loaders(args, train_transform, val_transform)

    losses = dict(
        cmg_loss=nn.MSELoss(reduction='sum'),
        img_loss=nn.L1Loss(reduction='sum')
    )

    output_transform = OutputReplaceTransformK()

    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag

    model = UnetKS(in_chans=data_chans, out_chans=data_chans, ext_chans=args.chans, chans=args.chans,
                   num_pool_layers=args.num_pool_layers, min_ext_size=args.min_ext_size, max_ext_size=args.max_ext_size,
                   use_ext_bias=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    trainer = ModelTrainerIMG(args, model, optimizer, train_loader, val_loader, output_transform, losses)
    trainer.train_model()


if __name__ == '__main__':
    settings = dict(
        # Variables that almost never change.
        challenge='multicoil',
        data_root='/media/veritas/D/FastMRI',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,  # This MUST be 1 for now.
        chans=32,
        num_pool_layers=4,
        save_best_only=True,
        pin_memory=False,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        smoothing_factor=8,

        # Variables that occasionally change.
        max_images=8,  # Maximum number of images to save.
        num_workers=3,
        init_lr=1E-3,
        gpu=1,  # Set to None for CPU mode.
        max_to_keep=0,
        img_lambda=1E-4,

        start_slice=10,
        min_ext_size=3,
        max_ext_size=15,

        # Variables that change frequently.
        sample_rate=1,
        num_epochs=10,
        verbose=False,
        use_slice_metrics=False,
        # prev_model_ckpt='',
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)
