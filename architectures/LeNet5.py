import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class Atanh(nn.Module):
    """Scaled tanH implementation.

    """
    def __init__(self, A=1.7159, S=2/3):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, x):
        return self.A * torch.tanh(self.S * x)


class SubSamplingLayer(nn.Module):
    """Sub-sampling module that performs a local averaging and a sub-sampling.
    
    """

    def __init__(self, in_channels=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels)

    def forward(self, x):
        batch, c, w, h = x.shape
        x = x.reshape(batch, c, w // 2, 2, h // 2, 2)
        x = torch.sum(x, dim=(3, 5))
        x = self.conv(x)
        return x


class C3Conv2DLayer(nn.Module):
    """C3 is a convolutional layer with 16 feature maps. The convolution kernel has a size of 5x5, an input size of
    14x14 and an output of 10x10. The feature maps in S2 are connected to C3 in a special way.

    """

    def __init__(self, in_channels=6, out_channels=16, kernel_size=5, stride=1, device='cpu'):
        super().__init__()
        self.connection_map = torch.tensor(
            [[1, 0, 0, 0, 1, 1,  1, 0, 0, 1, 1, 1,  1, 0, 1,  1],
             [1, 1, 0, 0, 0, 1,  1, 1, 0, 0, 1, 1,  1, 1, 0,  1],
             [1, 1, 1, 0, 0, 0,  1, 1, 1, 0, 0, 1,  0, 1, 1,  1],
             [0, 1, 1, 1, 0, 0,  1, 1, 1, 1, 0, 0,  1, 0, 1,  1],
             [0, 0, 1, 1, 1, 0,  0, 1, 1, 1, 1, 0,  1, 1, 0,  1],
             [0, 0, 0, 1, 1, 1,  0, 0, 1, 1, 1, 1,  0, 1, 1,  1]],
            dtype=torch.float32).to(device)
        # reshape the map
        self.connection_map = self.connection_map.transpose(1, 0).reshape(out_channels, in_channels, 1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        self.conv.weight.data = self.conv.weight.data * self.connection_map
        x = self.conv(x)
        return x



class LeNet5(nn.Module):
    """Implementation of the LeNet-5 from
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

    """

    def __init__(self, n_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Layer C1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            Atanh(),
            # Layer S2
            SubSamplingLayer(in_channels=6),
            nn.Sigmoid(),
            # Layer C3
            C3Conv2DLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            Atanh(),
            # Layer S4
            SubSamplingLayer(in_channels=16),
            # Layer C4
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            Atanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            Atanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
