from unittest import TestCase
from src.helpers import Device
import logging
import torch
logging.basicConfig(level=logging.DEBUG)

class TestDevice(TestCase):
    def test_get(self):
        # initialize the device without CUDA
        device = Device()
        self.assertIsInstance(device.get(), torch.device)


