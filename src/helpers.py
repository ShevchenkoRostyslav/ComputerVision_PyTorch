from collections import OrderedDict
from datetime import datetime
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt
import numpy as np

from src.metrics import Metric


class Device(object):
    def __init__(self, disable_cuda: bool=False):
        device = 'cpu'
        if not disable_cuda:
            logging.info(f'Trying to initialize CUDA device')
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                logging.info('No CUDA device found.')
        self.device = torch.device(device)
        logging.info(f'Device initialized with {device}')
    
    def get(self):
        return self.device


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')


def training_step(train_loader: DataLoader, model, criterion, optimizer: torch.optim.Optimizer, device: torch.device):
    """Training step of the training loop.

    :param train_loader:
    :param model: torch Model
    :param criterion:
    :param optimizer:
    :param device:
    :return: updated model, optimized, epoch_loss
    """
    # activate training mode
    model.train()
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y_true.to(device)

        # forward pass
        logits, _ = model(X)
        loss = criterion(logits, y_true)
        running_loss += loss.item() * X.size(0)

        # backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validation_step(valid_loader: DataLoader, model, criterion, device: torch.device):
    """Validation step of the training loop.

    :param valid_loader:
    :param model:
    :param criterion:
    :param device:
    :return: model, epoch_loss
    """
    # activate evaluation mode
    model.eval()
    running_loss = 0
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # forward pass
        logits, _ = model(X)
        loss = criterion(logits, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def train(model, criterion, optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          epochs: int,
          device: torch.device,
          eval_metrics: Sequence[Metric],
          print_every=1):
    """Training loop implementation.

    :param model:
    :param criterion:
    :param optimizer:
    :param train_loader:
    :param valid_loader:
    :param epochs:
    :param device:
    :param print_every:
    :return:
    """
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # training loop
    for epoch in range(epochs):
        # training step
        model, optimizer, train_loss = training_step(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        # validation step
        # # deactivate autograd
        with torch.no_grad():
            model, valid_loss = validation_step(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if print_every and epoch % print_every == print_every - 1:

            metrics_score = OrderedDict(
                {
                    **{metric.name: 0 for metric in eval_metrics},
                    **{metric.name+'_val': 0 for metric in eval_metrics},
                }
            )
            metric_output = ''
            # evaluation of the training
            for x_train, y_train in train_loader:
                _, y_pred_prob = model(x_train)
                _, y_pred = torch.max(y_pred_prob, 1)
                for metric in eval_metrics:
                    # metric is calculated PER BATCH -> to correct we multiply by the batch size and divide by the shape
                    metrics_score[metric.name] += metric(y_pred=y_pred, y_true=y_train)*y_pred.shape[0]
            # evaluation of the validation data
            for x_val, y_val in valid_loader:
                _, y_pred_prob = model(x_val)
                _, y_pred = torch.max(y_pred_prob, 1)
                for metric in eval_metrics:
                    # metric is calculated PER BATCH -> to correct we multiply by the batch size and divide by the shape
                    metrics_score[metric.name+'_val'] += metric(y_pred=y_pred, y_true=y_val)*y_pred.shape[0]

            for metric in eval_metrics:
                metrics_score[metric.name] = metrics_score[metric.name] / len(train_loader.dataset)
                metrics_score[metric.name+'_val'] = metrics_score[metric.name+'_val'] / len(valid_loader.dataset)
                metric_output = f'{metric.name}: {metrics_score[metric.name]:.4f}\tValidation {metric.name}: {metrics_score[metric.name+"_val"]:.4f}'

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'{metric_output}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)