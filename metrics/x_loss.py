import torch
from torch import nn, Tensor


class AlignmentLoss(nn.Module):
    """
    Implements the loss of the squared distance between two vectors in the two-channel format.
    The cosine law states that c^2 = a^2 + b^2 - 2*a*b*cos(C)
    where 'a' is the recon vector, 'b' is the target vector, 'C' is the angle between the two vectors,
    and 'c' is the vector between the endpoints of the two vectors 'a' and 'b'.

    Since 'b' is constant, b^2 is not necessary for minimization, although the output value will not be c^2 anymore.
    Therefore, a^2 - 2*a*b*cos(C) = a * (a - 2*b*cos(C)) is minimized.

    This is important since complex values are vectors in a complex coordinate system with real and imaginary axes.
    Minimizing the distance between the two vectors therefore minimizes both the phase and magnitude of the
    complex reconstruction and target values.

    P.S. The sign of C is irrelevant since cos(C) = cos(-C).
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, img_recon: Tensor, img_target: Tensor, phase_recon: Tensor, phase_target: Tensor) -> Tensor:
        loss = img_recon * (img_recon - 2 * img_target * torch.cos(phase_recon - phase_target))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction != 'none':
            raise NotImplementedError('Invalid reduction')
        return loss
