import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from utils.train_utils import create_custom_data_loaders

from train.subsample import MaskFunc
from data.input_transforms import Prefetch2Device, TrainPreProcessK
from data.output_transforms import OutputReplaceTransformK

from models.ase_unet import UnetASE
from train.model_trainers.model_trainer_K2CI import ModelTrainerK2CI
from metrics.custom_losses import CSSIM
from metrics.combination_losses import L1CSSIM7


"""
Memo: I have found that there is a great deal of variation in performance when training.
Even under the same settings, the results can be extremely different when using small numbers of samples. 
I believe that this is because of the large degree of variation in data quality in the dataset.
Therefore, demonstrating that one method works better than another requires using a large portion of the dataset.
However, this takes a lot of time...
Using small datasets for multiple runs may also prove useful.
"""


def train_img(args):

    # Maybe move this to args later.
    train_method = 'K2CI'

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

    input_train_transform = TrainPreProcessK(mask_func, args.challenge, args.device, use_seed=False, divisor=divisor)
    input_val_transform = TrainPreProcessK(mask_func, args.challenge, args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_custom_data_loaders(args, transform=data_prefetch)

    losses = dict(
        cmg_loss=nn.MSELoss(reduction='mean'),
        # img_loss=L1CSSIM7(reduction='mean', alpha=0.5)
        img_loss=CSSIM(filter_size=7, reduction='mean')
    )

    output_transform = OutputReplaceTransformK()

    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag

    model = UnetASE(in_chans=data_chans, out_chans=data_chans, ext_chans=args.chans, chans=args.chans,
                    num_pool_layers=args.num_pool_layers, min_ext_size=args.min_ext_size,
                    max_ext_size=args.max_ext_size, use_ext_bias=args.use_ext_bias, use_att=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_red_epoch, gamma=args.lr_red_rate)

    trainer = ModelTrainerK2CI(args, model, optimizer, train_loader, val_loader,
                               input_train_transform, input_val_transform, output_transform, losses, scheduler)

    trainer.train_model()

    # TODO: Maybe implement automatic deletion of empty log and checkpoint files
    #  that were created from failed or excessively short runs.


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
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        smoothing_factor=8,

        # Variables that occasionally change.
        chans=8,
        max_images=8,  # Maximum number of images to save.
        num_workers=1,
        init_lr=1E-3,
        gpu=0,  # Set to None for CPU mode.
        max_to_keep=0,
        start_slice=10,

        # Variables that change frequently.
        sample_rate=0.25,
        img_lambda=8,
        num_epochs=10,
        min_ext_size=1,
        max_ext_size=15,
        verbose=False,
        use_slice_metrics=True,  # Using slice metrics causes a 30% increase in training time.
        lr_red_epoch=20,
        lr_red_rate=0.1,
        use_ext_bias=True
        # prev_model_ckpt='',
    )
    options = create_arg_parser(**settings).parse_args()
    train_img(options)
