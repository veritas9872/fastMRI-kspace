import torch
from torch import nn, optim
from torchsummary import summary
from pathlib import Path
import random

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_custom_data_loaders, load_model_from_checkpoint

from train.subsample import MaskFunc, UniformMaskFunc, RandomMaskFunc
from data.input_transforms import Prefetch2Device, TrainPreProcessCC
from data.output_transforms import OutputTransformIK, MidTransformK

from models.fc_unet import FCUnet, Unet
from train.model_trainers.IMG_IK_original import ModelTrainerIMGIK

from metrics.custom_losses import logSSIMLoss, CSSIM


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
    divisor = 2 ** args.i_num_pool_layers

    mask_func = MaskFunc(args.center_fractions, args.accelerations)

    data_prefetch = Prefetch2Device(device)

    input_train_transform = TrainPreProcessCC(mask_func, args.challenge, args.device,
                                              use_seed=False, divisor=divisor)
    input_val_transform = TrainPreProcessCC(mask_func, args.challenge, args.device,
                                            use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_custom_data_loaders(args, transform=data_prefetch)

    losses = dict(
        cmg_loss=nn.MSELoss(reduction='mean'),
        img_loss=nn.MSELoss(reduction='mean'),
        ssim_loss=logSSIMLoss(reduction='mean')
    )

    mid_transform = MidTransformK()
    output_transform = OutputTransformIK()

    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag

    modelI = Unet(in_chans=data_chans, out_chans=data_chans, chans=args.i_chans,
                  num_pool_layers=args.i_num_pool_layers).to(device)
    modelK = Unet(in_chans=data_chans, out_chans=data_chans, chans=args.k_chans,
                  num_pool_layers=args.k_num_pool_layers).to(device)

    # Load pretrained model I parameters
    I_load_dir = './checkpoints/IMG/[IMG]GRUP2_SSIM2/ckpt_012.tar'
    load_model_from_checkpoint(modelI, I_load_dir, strict=True)

    # K_load_dir = './checkpoints/IMG/Trial 10  2019-08-15 21-01-57/ckpt_K014.tar'
    # load_model_from_checkpoint(modelK, K_load_dir, strict=True)

    # optimizer = optim.Adam(list(modelI.parameters()) + list(modelK.parameters()), lr=args.init_lr)
    optimizer = optim.Adam(modelK.parameters(), lr=args.init_lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)

    trainer = ModelTrainerIMGIK(args, modelI, modelK, optimizer, train_loader, val_loader, input_train_transform,
                                input_val_transform, mid_transform, output_transform, losses, scheduler)

    # TODO: Implement logging of model, losses, transforms, etc.
    trainer.train_model()


if __name__ == '__main__':
    settings = dict(
        # Variables that almost never change.
        name='IK_ssim_holdI',  # Please do change this every time Harry
        challenge='multicoil',
        data_root='/media/harry/fastmri/fastMRI_data',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,  # This MUST be 1 for now.
        i_chans=64,
        k_chans=32,
        i_num_pool_layers=5,
        k_num_pool_layers=5,
        save_best_only=False,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        smoothing_factor=8,

        # Variables that occasionally change.
        display_images=10,  # Maximum number of images to save.
        num_workers=4,
        init_lr=1e-4,
        gpu=0,  # Set to None for CPU mode.
        max_to_keep=1,
        img_lambda=3,
        ssim_lambda=10,

        start_slice=0,
        start_val_slice=0,

        # Variables that change frequently.
        sample_rate=1,
        num_epochs=100,
        verbose=False,
        use_slice_metrics=True,  # Using slice metrics causes a 30% increase in training time.
        lr_red_epoch=50,
        lr_red_rate=0.1,
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)