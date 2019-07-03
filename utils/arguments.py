import argparse


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Data parameters
        self.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                          help='Which challenge')

        self.add_argument('--data-root', type=str, required=True,
                          help='Path to the dataset')

        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        # Mask parameters
        self.add_argument('--accelerations', nargs='+', default=[4, 8], type=int,
                          help='Ratio of k-space columns to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for '
                               'each volume.')

        self.add_argument('--center-fractions', nargs='+', default=[0.08, 0.04], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')

        # Training parameters
        self.add_argument('--img-lambda', type=float, required=True,
                          help='The ratio between complex image loss and real image loss.'
                               'This parameter will probably have to be replaced soon. It is a temporary fix.')

        self.add_argument('--num-epochs', type=int, required=True,
                          help='Number of epochs to train the data.')

        self.add_argument('--init-lr', type=float, default=0.001,
                          help='Initial learning rate. The learning rate may be reduced by optimizers and schedulers.')

        self.add_argument('--chans', type=int, default=32,
                          help='The number of beginning channels used for the Unet model.')

        self.add_argument('--num-pool-layers', type=int, default=4,
                          help='Number of pooling layers in the Unet model.')

        self.add_argument('--start-slice', type=int, default=0,
                          help='The starting index of blocks to sample from. '
                               'This is useful for removing the frontal slices, which tend to be noisy.'
                               'Also considering percentile implementation later.')

        # DataLoader parameters
        self.add_argument('--pin-memory', type=bool, default=False,
                          help='Whether to use pinned memory for the DataLoader. '
                               'Not possible when data has been pre-fetched to GPU.')

        self.add_argument('--num-workers', type=int, default=1, help='Number of processes to use to prepare data.')

        self.add_argument('--batch-size', type=int, default=1, help='Batch size for input data.')

        self.add_argument('--gpu', type=int, default=None, help='Which GPU to use. Set to None to run on CPU')

        # Visualization and logging parameters.
        self.add_argument('--log-root', type=str, default='./logs',
                          help='Root log directory for saving Tensorboard outputs, log files, and more.')

        self.add_argument('--ckpt-root', type=str, default='./checkpoints',
                          help='Root directory for checkpoints of models.')

        self.add_argument('--max-to-keep', type=int, default=0,
                          help='Maximum number of model checkpoints to save during training.')

        self.add_argument('--verbose', type=bool, default=False,
                          help='Whether to display step-wise losses and metrics.')

        self.add_argument('--use-slice-metrics', type=bool, default=False,
                          help='Whether to calculate slice metrics on reconstructed images.')

        self.add_argument('--save-best-only', type=bool, default=False,
                          help='Whether to save checkpoints when the validation loss, or some other metric, improves')

        self.add_argument('--max-images', type=int, default=0,
                          help='Maximum number of validation images to display on TensorBoard during training.')

        self.add_argument('--smoothing-factor', type=float, default=8,
                          help='A number that decides how much to smooth the peaks of k-space visualization.'
                               'A larger number makes k-space peaks less pronounced, '
                               'making the rest of k-space more visible.')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)
