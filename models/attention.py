from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_chans, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling.
        self.gmp = nn.AdaptiveMaxPool2d(1)  # Global Maximum Pooling.

        self.use_gap = use_gap
        self.use_gmp = use_gmp

        self.layer = nn.Sequential(
            nn.Linear(in_features=num_chans, out_features=num_chans // reduction),
            nn.ReLU(),
            nn.Linear(in_features=num_chans // reduction, out_features=num_chans)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        batch, chans, _, _ = tensor.shape
        if self.use_gap and self.use_gmp:
            gap = self.gap(tensor).view(batch, chans)
            gmp = self.gmp(tensor).view(batch, chans)
            # Maybe batch-norm the two pooling types to make their scales more similar.
            # This might make training slower, however.
            features = self.layer(gap) + self.layer(gmp)
            att = self.sigmoid(features).view(batch, chans, 1, 1)

        elif self.use_gap:
            gap = self.gap(tensor).view(batch, chans)
            features = self.layer(gap)
            att = self.sigmoid(features).view(batch, chans, 1, 1)

        elif self.use_gmp:
            gmp = self.gmp(tensor).view(batch, chans)
            features = self.layer(gmp)
            att = self.sigmoid(features).view(batch, chans, 1, 1)

        else:
            att = 1

        return tensor * att
