import torch
from torch import nn, optim
from torchsummary import summary
from pathlib import Path
import random

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_custom_data_loaders, load_model_from_checkpoint

from train.subsample import MaskFunc, UniformMaskFunc, RandomMaskFunc
from data.input_transforms import Prefetch2Device, TrainPreProcessIMG, TrainPreProcessHC
from data.output_transforms import OutputTransformCC

from models.fc_unet import FCUnet, Unet
from models.edsr_unet import UNet
from train.model_trainers.IMG_trainer import ModelTrainerIMG

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
    divisor = 2 ** args.num_pool_layers

    mask_func = MaskFunc(args.center_fractions, args.accelerations)

    data_prefetch = Prefetch2Device(device)

    input_train_transform = TrainPreProcessHC(mask_func, args.challenge, args.device,
                                              use_seed=False, divisor=divisor)
    input_val_transform = TrainPreProcessHC(mask_func, args.challenge, args.device,
                                            use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_custom_data_loaders(args, transform=data_prefetch)

    losses = dict(
        cmg_loss=nn.MSELoss(reduction='mean'),
        img_loss=nn.MSELoss(reduction='mean'),
        consistency_loss=nn.MSELoss(reduction='mean'),
        ssim_loss=logSSIMLoss(reduction='mean')
    )
    output_transform = OutputTransformCC()

    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag

    model = Unet(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
                 num_pool_layers=args.num_pool_layers).to(device)
    # model = UNet(in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
    #              num_depth_blocks=2).to(device)

    # Load pretrained model parameters
    load_dir = './checkpoints/IMG/[IMG]GRUP2_SSIM2/ckpt_012.tar'
    load_model_from_checkpoint(model, load_dir, strict=True)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)

    trainer = ModelTrainerIMG(args, model, optimizer, train_loader, val_loader,
                              input_train_transform, input_val_transform, output_transform, losses, scheduler)

    # TODO: Implement logging of model, losses, transforms, etc.
    trainer.train_model()


if __name__ == '__main__':
    settings = dict(
        # Variables that almost never change.
        name = 'HC_image',  # Please do change this every time Harry
        challenge='multicoil',
        data_root='/media/harry/fastmri/fastMRI_data',
        log_root='./logs',
        ckpt_root='./checkpoints',
        batch_size=1,  # This MUST be 1 for now.
        chans=64,
        num_pool_layers=5,
        save_best_only=True,
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        smoothing_factor=8,

        # Variables that occasionally change.
        display_images=10,  # Maximum number of images to save.
        num_workers=4,
        init_lr=1e-4,
        gpu=0,  # Set to None for CPU mode.
        max_to_keep=1,
        img_lambda=5,
        consistency_lambda=5,
        ssim_lambda=10,

        start_slice=0,
        start_val_slice=0,

        # Variables that change frequently.
        sample_rate=1,
        num_epochs=150,
        verbose=False,
        use_slice_metrics=True,  # Using slice metrics causes a 30% increase in training time.
        lr_red_epoch=30,
        lr_red_rate=0.2,
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)