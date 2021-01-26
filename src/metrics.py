from typing import Optional, Union

import torch
from abc import ABCMeta, abstractmethod

__all__ = ['Metric']


class Metric(metaclass=ABCMeta):

    def __init__(self, is_multilabel: bool = False, device: Optional[Union[str, torch.device]] = None):
        self._device = device
        self._is_multilabel = is_multilabel
        self._type = None
        self._num_classes = None
        self.name = None

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pass

    def _check_type(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        if y_true.ndimension() + 1 == y_pred.ndimension() or y_true.ndimension() == y_pred.ndimension():
            num_classes = 1 if len(y_pred.shape) == 1 else y_pred.shape[1]
            if num_classes == 1:
                update_type = "binary"
            else:
                update_type = "multiclass"
        else:
            raise RuntimeError(
                f"Invalid shapes of y_true (shape={y_true.shape}) and y_pred (shape={y_pred.shape})"
            )
        if self._is_multilabel:
            update_type = "multilabel"

        self._type = update_type
        self._num_classes = num_classes

    @staticmethod
    def _check_shape(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        if not (y_true.ndimension() + 1 == y_pred.ndimension() or y_true.ndimension() == y_pred.ndimension()):
            raise ValueError(
                "y_true must have shape of (batch_size, ...) and y_pred must have "
                "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                f"but given {y_true.shape} vs {y_pred.shape}."
            )


class Accuracy(Metric):
    """Calculates the accuracy for binary, multiclass and multilabel data.

    """
    
    def __init__(self, is_multilabel: bool = False, device: Optional[Union[str, torch.device]] = None):
        super().__init__(is_multilabel, device)
        self.name = 'Accuracy'

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self._check_type(y_pred=y_pred, y_true=y_true)
        self._check_shape(y_pred=y_pred, y_true=y_true)

        n = y_pred.shape[0]
        if self._type == 'binary':
            correct = (y_pred == y_true).sum()

        return correct / n


