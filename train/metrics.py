import torch.nn as nn
import torch.nn.functional as F


class CustomL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()

        if reduction == 'sum':
            self.sum_outputs = True
        elif reduction == 'sum':
            self.sum_outputs = False
        else:
            raise ValueError('Invalid reduction type')

    def forward(self, image_recons, targets):
        """

        Args:
            image_recons (list):
            targets (list):

        Returns:

        """
        # assert len(image_recons) == len(targets)  # This is not possible when the input is a generator.

        if self.sum_outputs:
            return sum(F.l1_loss(recon, target, reduction='sum') for recon, target in zip(image_recons, targets))
        else:
            return sum(F.l1_loss(recon, target, reduction='mean')
                       for recon, target in zip(image_recons, targets)) / len(targets)

