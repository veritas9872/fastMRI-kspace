from torch.utils.data import DataLoader

from pathlib import Path

from data.mri_data import CustomSliceData
from data.input_transforms import Prefetch2Device


def temp_collate_fn(batch):
    return batch[0]


def create_prefetch_datasets(args):
    transform = Prefetch2Device(device=args.device)

    arguments = vars(args)  # Placed here for backward compatibility and convenience.
    args.sample_rate_train = arguments.get('sample_rate_train', arguments.get('sample_rate'))
    args.sample_rate_val = arguments.get('sample_rate_val', arguments.get('sample_rate'))
    args.start_slice_train = arguments.get('start_slice_train', arguments.get('start_slice'))
    args.start_slice_val = arguments.get('start_slice_val', arguments.get('start_slice'))

    # Generating Datasets.
    train_dataset = CustomSliceData(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate_train,
        start_slice=args.start_slice_train,
        use_gt=args.use_gt
    )

    val_dataset = CustomSliceData(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=transform,
        challenge=args.challenge,
        sample_rate=args.sample_rate_val,
        start_slice=args.start_slice_val,
        use_gt=args.use_gt
    )
    return train_dataset, val_dataset


def create_prefetch_data_loaders(args):
    train_dataset, val_dataset = create_prefetch_datasets(args)

    if args.batch_size > 1:
        raise NotImplementedError('Batch size should be 1 for now.')

    collate_fn = temp_collate_fn

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader
