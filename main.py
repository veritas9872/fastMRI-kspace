import torch
from torch import nn, optim

from pathlib import Path

from utils.run_utils import initialize, save_dict_as_json, get_logger, create_arg_parser
from data.pre_processing import KInputSliceTransform, InputSliceTransformK2C, \
    WeightedInputSliceK2K, WeightedInputSliceK2C, InputSliceTransformC2C, InputSliceTransformK2K
from data.post_processing import OutputReplaceTransformK2C, \
    WeightedOutputReplaceK2K, WeightedOutputReplaceK2C, OutputTransformC2C, OutputBatchReplaceTransformK2K
from utils.train_utils import create_data_loaders
from train.subsample import MaskFunc
from train.processing import SingleBatchOutputTransform, OutputBatchTransform
from train.model_trainers.model_trainer_K2I import ModelTrainerK2I
from train.model_trainers.model_trainer_K2K import ModelTrainerK2K
from train.model_trainers.model_trainer_K2C import ModelTrainerK2C
from train.model_trainers.model_trainer_C2C import ModelTrainerC2C
from train.metrics import CustomL1Loss
from models.unet_model import NewUnetModel
from models.new_unet_model import Unet
from models.ksse_unet import UnetKSSE


# Try out SSIM and MS-SSIM as loss functions. They appear to be effective in getting fine-grained features,
# unlike L1.


def train_k2i():
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=0.01,  # Mostly for debugging purposes.
        num_workers=2,
        init_lr=1E-3,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=2,
        max_to_keep=1,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/F/lzfCompFastMRI',  # Using compressed dataset for better I/O performance.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_images=1,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
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

    input_slice_train_transform = KInputSliceTransform(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=False, divisor=divisor)

    input_slice_val_transform = KInputSliceTransform(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(
        args=args, train_transform=input_slice_train_transform, val_transform=input_slice_val_transform)

    # Loss Function and output post-processing functions.
    if args.batch_size == 1:
        loss_func = nn.L1Loss(reduction='mean')
        output_batch_transform = SingleBatchOutputTransform()
    elif args.batch_size > 1:
        loss_func = CustomL1Loss(reduction='mean')
        output_batch_transform = OutputBatchTransform()
    else:
        raise RuntimeError('Invalid batch size.')

    # Define model.
    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
    model = NewUnetModel(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
                         num_pool_layers=args.num_pool_layers).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    trainer = ModelTrainerK2I(args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                              post_processing=output_batch_transform, loss_func=loss_func)

    trainer.train_model()


def train_k2k(args):
    if args.batch_size > 1:
        raise NotImplementedError('K2K for batch size greater than 1 not implemented yet.')

    # Move this to args later.
    train_method = 'K2K'

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / train_method
    ckpt_path.mkdir(exist_ok=True)
    
    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
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

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    # Create a function for saving such things later.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Important!: I need to log everything that I use: model, pre-processing, loss function for each domain,
    # model trainer, etc. properly if I am going to use this stuff properly later on.

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

    input_slice_train_transform = InputSliceTransformK2K(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=False, divisor=divisor)

    input_slice_val_transform = InputSliceTransformK2K(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(
        args=args, train_transform=input_slice_train_transform, val_transform=input_slice_val_transform)

    # Loss Function and output post-processing functions.
    if args.batch_size == 1:
        loss_func = nn.MSELoss(reduction='sum')
        output_batch_transform = OutputBatchReplaceTransformK2K()  # Send log_scale to args later.
    else:
        raise RuntimeError('Invalid batch size.')

    # Define model.
    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
    model = UnetKSSE(in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
                     min_ext_size=1, max_ext_size=17, use_ext_bias=True, use_res_out=False).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    trainer = ModelTrainerK2K(args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                              post_processing=output_batch_transform, k_loss=loss_func)

    trainer.train_model()


def train_k2c(args):
    if args.batch_size > 1:
        raise NotImplementedError('K2C for batch size greater than 1 not implemented yet.')

    # Maybe move this to args later.
    train_method = 'K2C'

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / train_method
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
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

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    # Create a function for saving such things later.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Important!: I need to log everything that I use: model, pre-processing, loss function for each domain,
    # model trainer, etc. properly if I am going to use this stuff properly later on.

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

    if args.apply_weighting:
        input_slice_train_transform = WeightedInputSliceK2C(
            mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=False, divisor=divisor)

        input_slice_val_transform = WeightedInputSliceK2C(
            mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=True, divisor=divisor)
    else:
        input_slice_train_transform = InputSliceTransformK2C(
            mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=False, divisor=divisor)

        input_slice_val_transform = InputSliceTransformK2C(
            mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(
        args=args, train_transform=input_slice_train_transform, val_transform=input_slice_val_transform)

    # Loss Function and output post-processing functions.
    if args.batch_size == 1:
        loss_func = nn.MSELoss(reduction='sum')
        if args.apply_weighting:
            output_batch_transform = WeightedOutputReplaceK2C()
        else:
            output_batch_transform = OutputReplaceTransformK2C()
    else:
        raise RuntimeError('Invalid batch size.')

    # Define model.
    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
    # model = NewUnetModel(in_chans=data_chans, out_chans=data_chans, chans=args.chans,
    #                      num_pool_layers=args.num_pool_layers).to(device)
    model = UnetKSSE(in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
                     min_ext_size=3, max_ext_size=9, use_ext_bias=True).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    trainer = ModelTrainerK2C(args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                              post_processing=output_batch_transform, c_loss=loss_func)

    trainer.train_model()


def train_c2c():
    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=1,  # Mostly for debugging purposes.
        num_workers=2,
        init_lr=1E-3,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=50,
        max_to_keep=1,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/D/FastMRI',  # Using compressed dataset for better I/O performance and chunking.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_images=8,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        pin_memory=False,
        add_graph=False,
        prev_model_ckpt='',
        apply_weighting=False,
        affine=True,
        residual=True
    )

    # Replace with a proper argument parsing function later.
    args = create_arg_parser(**defaults).parse_args()

    if args.batch_size > 1:
        raise NotImplementedError('C2C for batch size greater than 1 not implemented yet.')

    # Maybe move this to args later.
    train_method = 'C2C'

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / train_method
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
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

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    # Create a function for saving such things later.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Important!: I need to log everything that I use: model, pre-processing, loss function for each domain,
    # model trainer, etc. properly if I am going to use this stuff properly later on.

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

    input_slice_train_transform = InputSliceTransformC2C(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=False, divisor=divisor)

    input_slice_val_transform = InputSliceTransformC2C(
        mask_func=mask_func, challenge=args.challenge, device=args.device, use_seed=True, divisor=divisor)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(
        args=args, train_transform=input_slice_train_transform, val_transform=input_slice_val_transform)

    # Loss Function and output post-processing functions.
    loss_func = nn.MSELoss(reduction='sum')
    output_batch_transform = OutputTransformC2C()

    # Define model.
    data_chans = 2 if args.challenge == 'singlecoil' else 30  # Multicoil has 15 coils with 2 for real/imag
    model = Unet(in_chans=data_chans, out_chans=data_chans, chans=args.chans, num_pool_layers=args.num_pool_layers,
                 affine=args.affine, residual=args.residual).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    trainer = ModelTrainerC2C(args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                              post_processing=output_batch_transform, c_loss=loss_func)

    trainer.train_model()


if __name__ == '__main__':
    # defaults = dict(
    #     batch_size=1,  # This MUST be 1 for now.
    #     sample_rate=1,  # Mostly for debugging purposes.
    #     num_workers=3,
    #     init_lr=1E-3,
    #     log_dir='./logs',
    #     ckpt_dir='./checkpoints',
    #     gpu=0,  # Set to None for CPU mode.
    #     num_epochs=10,
    #     max_to_keep=1,
    #     verbose=False,
    #     save_best_only=True,
    #     data_root='/media/veritas/D/FastMRI',  # Using compressed dataset for better I/O performance.
    #     challenge='multicoil',
    #     center_fractions=[0.08, 0.04],
    #     accelerations=[4, 8],
    #     max_images=8,  # Maximum number of images to save.
    #     chans=32,
    #     num_pool_layers=4,
    #     pin_memory=False,
    #     add_graph=False
    # )

    defaults = dict(
        batch_size=1,  # This MUST be 1 for now.
        sample_rate=1,  # Mostly for debugging purposes.
        num_workers=2,
        init_lr=1E-3,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=30,
        max_to_keep=1,
        verbose=False,
        save_best_only=True,
        data_root='/media/veritas/D/FastMRI',  # Using compressed dataset for better I/O performance and chunking.
        challenge='multicoil',
        center_fractions=[0.08, 0.04],
        accelerations=[4, 8],
        max_images=8,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        pin_memory=False,
        add_graph=False,
        prev_model_ckpt='',
        apply_weighting=False
    )

    # Replace with a proper argument parsing function later.
    arg = create_arg_parser(**defaults).parse_args()
    train_k2c(arg)
