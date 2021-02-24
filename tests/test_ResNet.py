from unittest import TestCase
import torch
from torchsummary import summary
from architectures.ResNet import *


class TestConv2dSamePad(TestCase):

    def test_create(self):
        with self.subTest('case 1'):
            layer = Conv2dSamePad(in_channels=32, out_channels=64, kernel_size=3, stride=1)
            # (32 - 3 + 2P)/1 + 1 = 32 -> P=1
            self.assertEqual(layer.padding, (1, 1))

        with self.subTest('case 2'):
            layer = Conv3x3SamePad(in_channels=1, out_channels=3)
            torch.nn.init.constant_(layer.weight, 5.)
            dummy_input = torch.ones([1, 1, 1, 1])
            output = layer(dummy_input)
            expected_output = torch.tensor([5., 5., 5.]).view([1, 3, 1, 1])
            self.assertTrue(torch.equal(expected_output, output))


class TestResNetResidualLayer(TestCase):

    def test_create(self):
        with self.subTest('case 1'):
            layer = ResNetResidualLayer(64, 128, n=3, activation='selu')
            # torch.nn.init.constant_(layer.blocks.weight, 5.)
            dummy_input = torch.ones([1, 64, 48, 48])
            output = layer(dummy_input)
            print(output.shape)


class TestResidualBlock(TestCase):

    def test_create(self):
        with self.subTest('case 1'):
            layer = ResidualBlock(3, 32, activation='relu')
            dummy_input = torch.ones([1, 3, 32, 32])
            output = layer(dummy_input)
            expected_output = torch.ones([1, 3, 32, 32]) * 2
            self.assertTrue(torch.equal(expected_output, output))


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.constant_(m.weight, 5)
    elif 'BatchNorm' in classname:
        torch.nn.init.constant_(m.weight, 5)


class TestResNetResidualBlock(TestCase):

    def test_create(self):
        with self.subTest('case 1'):
            layer = ResNetResidualBlock(2, 2, 2, 'relu', downsampling_stride=1)
            layer.apply(weights_init)
            dummy_input = torch.ones([1, 2, 32, 32])
            output = layer(dummy_input)
            self.assertEqual(output.shape, dummy_input.shape)

        with self.subTest('case 2'):
            layer = ResNetResidualBlock(2, 4, 2, 'relu', downsampling_stride=2)
            layer.apply(weights_init)
            dummy_input = torch.ones([1, 2, 32, 32])
            output = layer(dummy_input)
            expected_output_shape = [1, 4, 16, 16]
            self.assertEqual(list(output.shape), expected_output_shape)

class TestResNet(TestCase):

    def test_create(self):
        model = resnet18(3, 1000)
        summary(model, (3, 224, 224))