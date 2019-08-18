import torch
from torch import nn, optim

from pathlib import Path


class GANCheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, modelG, modelD, optimizerG, optimizerD,
                 mode='min', save_best_only=True, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelD, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizerD, optim.Optimizer), 'Not a Pytorch Optimizer'
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        assert mode in ('min', 'max'), 'Mode must be either `min` or `max`'
        ckpt_path = Path(ckpt_dir)
        assert ckpt_path.exists(), 'Not a valid, existing path'

        record_path = ckpt_path / 'Checkpoints.txt'

        try:
            record_file = open(record_path, mode='x')
        except FileExistsError:
            import sys
            print('WARNING: It is recommended to have a separate checkpoint directory for each run.', file=sys.stderr)
            print('Appending to previous Checkpoint record file!', file=sys.stderr)
            record_file = open(record_path, mode='a')

        print(f'Checkpoint List for {ckpt_path}', file=record_file)
        record_file.close()

        self.modelG = modelG
        self.modelD = modelD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.save_best_only = save_best_only
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.save_counter = 0
        self.record_path = record_path
        self.record_dict = dict()

        if mode == 'min':
            self.prev_best = float('inf')
            self.mode = mode
        elif mode == 'max':
            self.prev_best = -float('inf')
            self.mode = mode
        else:
            raise TypeError('Mode must be either `min` or `max`')

    def _save(self, ckpt_name=None, **save_kwargs):
        self.save_counter += 1
        save_dictG = {'model_state_dict': self.modelG.state_dict(),
                      'optimizer_state_dict': self.optimizerG.state_dict()}
        save_dictD = {'model_state_dict': self.modelD.state_dict(),
                      'optimizer_state_dict': self.optimizerD.state_dict()}
        save_dictG.update(save_kwargs)
        save_dictD.update(save_kwargs)
        save_pathG = self.ckpt_path / (f'ckpt_G{self.save_counter:03d}.tar')
        save_pathD = self.ckpt_path / (f'ckpt_D{self.save_counter:03d}.tar')

        torch.save(save_dictG, save_pathG)
        torch.save(save_dictD, save_pathD)
        print(f'Saved Checkpoint to {save_pathG}{save_pathD}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathG} {save_pathD}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathG} {save_pathD}', file=file)

        self.record_dict[self.save_counter] = save_pathG

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathG, save_pathD

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_pathG = None
        if is_best or not self.save_best_only:
            save_pathG, save_pathD = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathG, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dir, load_optimizer=True):
        save_dict = torch.load(load_dir)

        self.model.load_state_dict(save_dict['model_state_dict'])
        print(f'Loaded model parameters from {load_dir}')

        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dir}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


class IKCheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, modelI, modelK, optimizer,
                 mode='min', save_best_only=False, ckpt_dir='./checkpoints', max_to_keep=5):

        # Type checking.
        assert isinstance(modelI, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizer, optim.Optimizer), 'Not a Pytorch Optimizer'
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        assert mode in ('min', 'max'), 'Mode must be either `min` or `max`'
        ckpt_path = Path(ckpt_dir)
        assert ckpt_path.exists(), 'Not a valid, existing path'

        record_path = ckpt_path / 'Checkpoints.txt'

        try:
            record_file = open(record_path, mode='x')
        except FileExistsError:
            import sys
            print('WARNING: It is recommended to have a separate checkpoint directory for each run.', file=sys.stderr)
            print('Appending to previous Checkpoint record file!', file=sys.stderr)
            record_file = open(record_path, mode='a')

        print(f'Checkpoint List for {ckpt_path}', file=record_file)
        record_file.close()

        self.modelI = modelI
        self.modelK = modelK
        self.optimizer = optimizer
        self.save_best_only = save_best_only
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.save_counter = 0
        self.record_path = record_path
        self.record_dict = dict()

        if mode == 'min':
            self.prev_best = float('inf')
            self.mode = mode
        elif mode == 'max':
            self.prev_best = -float('inf')
            self.mode = mode
        else:
            raise TypeError('Mode must be either `min` or `max`')

    def _save(self, ckpt_name=None, **save_kwargs):
        self.save_counter += 1
        save_dictI = {'model_state_dict': self.modelI.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}
        save_dictK = {'model_state_dict': self.modelK.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}
        save_dictI.update(save_kwargs)
        save_dictK.update(save_kwargs)
        save_pathI = self.ckpt_path / (f'ckpt_I{self.save_counter:03d}.tar')
        save_pathK = self.ckpt_path / (f'ckpt_K{self.save_counter:03d}.tar')

        torch.save(save_dictI, save_pathI)
        torch.save(save_dictK, save_pathK)
        print(f'Saved Checkpoint to {save_pathI} {save_pathK}')
        print(f'Checkpoint {self.save_counter:04d}: {save_pathI} {save_pathK}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_pathI} {save_pathK}', file=file)

        self.record_dict[self.save_counter] = save_pathK

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_pathI, save_pathK

    def save(self, metric, verbose=True, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_pathK = None
        if is_best or not self.save_best_only:
            save_pathI, save_pathK = self._save(ckpt_name, **save_kwargs)

        if verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_pathK, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dir, load_optimizer=True):
        save_dict = torch.load(load_dir)

        self.model.load_state_dict(save_dict['model_state_dict'])
        print(f'Loaded model parameters from {load_dir}')

        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dir}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


def load_gan_model_from_checkpoint(model, load_dir, strict=False):
    """
    A simple function for loading checkpoints without having to use Checkpoint Manager. Very useful for evaluation.
    Checkpoint manager was designed for loading checkpoints before resuming training.

    model (nn.Module): Model architecture to be used.
    load_dir (str): File path to the checkpoint file. Can also be a Path instead of a string.
    """
    assert isinstance(model, nn.Module), 'Model must be a Pytorch module.'
    assert Path(load_dir).exists(), 'The specified directory does not exist'
    save_dict = torch.load(load_dir)
    model.load_state_dict(save_dict['model_state_dict'], strict=strict)
    return model  # Not actually necessary to return the model but doing so anyway.