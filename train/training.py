import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import runstats
from tqdm import tqdm

from pathlib import Path
from time import time

from utils.run_utils import get_logger, initialize, save_dict_as_json
from utils.train_utils import CheckpointManager
from train.subsample import MaskFunc
from data.mri_data import SliceData
from data.data_transforms import DataTrainTransform
from models.k_unet_model import UnetModel


def create_datasets(args):
    train_mask_func = MaskFunc(args.center_fractions, args.accelerations)
    val_mask_func = MaskFunc(args.center_fractions, args.accelerations)

    train_dataset = SliceData(
        root=args.data_path / f'{args.challenge}_train',
        transform=DataTrainTransform(train_mask_func, args.resolution, args.challenge, use_seed=False),
        challenge=args.challenge
    )

    val_dataset = SliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTrainTransform(val_mask_func, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
    )

    return train_dataset, val_dataset


def create_data_loaders(args):
    train_dataset, val_dataset = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_step():
    pass


def train_epoch():
    pass


def val_step():
    pass


def val_epoch():
    pass


def train_model(args):
    ckpt_path = Path('checkpoints')
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    args.ckpt_path = ckpt_path

    log_path = Path('logs')
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    args.log_path = log_path

    save_dict_as_json(vars(save_dict_as_json), log_dir=log_path, save_name=run_name)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    if not runstats.__compiled__:
        logger.warning('runstats module is not cython compiled.')

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        args.device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Create Datasets. Use one slice at a time for now.
    train_loader, val_loader = create_data_loaders(args)

    # Define model.
    model = UnetModel(in_chans=2, out_chans=1, chans=args.chans, num_pool_layers=args.num_pool_layers,
                      drop_prob=0.).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    loss_func = nn.L1Loss(reduction='mean').to(args.device)

    checkpointer = CheckpointManager(model, optimizer, args.save_best_only, ckpt_path, args.max_to_keep)

    if hasattr(args, 'previous_model') and args.previous_model:
        checkpointer.load(load_dir=checkpointer, load_optimizer=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True, cooldown=0, min_lr=1E-7)

    writer = SummaryWriter(log_dir=str(log_path))

    example_inputs = torch.ones(size=(args.batch_size, 2, 320, 320)).to(args.device, non_blocking=True)
    writer.add_graph(model=model, input_to_model=example_inputs, verbose=False)

    previous_best = float('inf')

    logger.info('Beginning Training loop')
    for epoch in range(1, args.num_epochs + 1):  # 1 based indexing
        # Training
        tic = time()
        # train_loss_sum = train_epoch()
        toc = int(time() - tic)
        # train_epoch_loss

        # Evaluating
        tic = time()
        # val_loss_sum = val_epoch()
        toc = int(time() - tic)
        # val_epoch_loss

