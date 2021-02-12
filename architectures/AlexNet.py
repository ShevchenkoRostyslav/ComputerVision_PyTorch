import torch

from torch import nn
import torch.nn.functional as F


class LocalResponceNorm(nn.Module):
    """Local Response Normalization Layer - square-normalizes the pixel values in a feature map
    within a local neighborhood.
    Now implemented natively in pytorch via nn.LocalResponseNorm
    explained:
    https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac

    :param k: additive factor
    :param n:  amount of neighbouring pixels used for normalization
    :param alpha: multiplicative factor
    :param beta: exponent
    :return:
    """

    def __init__(self, k=2, n=5, alpha=1e-04, beta=0.75, across_channels=False):
        super().__init__()
        self.across_channels = across_channels
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta
        if across_channels:
            self.average = nn.AvgPool3d(kernel_size=(self.n, 1, 1), stride=1, padding=(int((self.n-1.0)/2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=self.n, stride=1, padding=int((self.n - 1.0) / 2))

    def forward(self, x):
        x = x / (self.k + self.alpha * self.average(x ** 2)) ** self.beta
        return x


class OverlappingPool(nn.MaxPool2d):
    """Overlapping Pooling Layer - max pooling with a stride < the kernel size.

    """

    def __init__(self):
        super().__init__(kernel_size=3, stride=2)


class AlexNet(nn.Module):
    """Implementation of the AlexNet from:
    "ImageNet Classification with Deep Convolutional Neural Networks", A. Krizhevsky et al.
    https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    """

    def __init__(self, n_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Layer C1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2), # -> (224-11+2*2)/4 + 1 = 55
            nn.ReLU(inplace=True),
            LocalResponceNorm(),
            OverlappingPool(), # -> (55 - 3)/2 + 1 = 27
            # Layer C2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), # 27
            nn.ReLU(inplace=True),
            LocalResponceNorm(),
            OverlappingPool(), # -> (27 - 3)/2 + 1 = 13
            # Layer C3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), # -> 13
            nn.ReLU(inplace=True),
            # Layer C4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # -> 13
            nn.ReLU(inplace=True),
            # Layer C5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), # -> 13
            nn.ReLU(inplace=True),
            OverlappingPool() # -> 6
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
