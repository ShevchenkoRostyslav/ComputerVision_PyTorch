from unittest import TestCase
from torch.nn import functional as f
import torch

from architectures.LeNet5 import SubSamplingLayer


class TestSubSamplingLayer(TestCase):

    def test_init(self):
        layer = SubSamplingLayer()
        x = torch.Tensor([1,2,3,4]).view(2,2)
        print(x)
        print(x.unfold(0, 1, 1))
