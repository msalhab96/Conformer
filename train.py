import os
import torch
from hprams import hprams
from functools import wraps
from torch.nn import Module
from data import DataLoader
from torch.optim import Optimizer
from typing import Callable
from tqdm import tqdm

MODEL_KEY = 'model'
OPTIMIZER_KEY = 'optimizer'


def save_checkpoint(func) -> Callable:
    """Save a checkpoint after each iteration
    """
    @wraps(func)
    def wrapper(obj, *args, _counter=[0], **kwargs):
        _counter[0] += 1
        result = func(obj, *args, **kwargs)
        if not os.path.exists(hprams.training.checkpoints_dir):
            os.mkdir(hprams.training.checkpoints_dir)
        checkpoint_path = os.path.join(
            hprams.training.checkpoints_dir,
            'checkpoint_' + str(_counter[0]) + '.pt'
            )
        torch.save(
            {
                MODEL_KEY: obj.model.state_dict(),
                OPTIMIZER_KEY: obj.optimizer.state_dict(),
            },
            checkpoint_path
            )
        print(f'checkpoint saved to {checkpoint_path}')
        return result
    return wrapper


class Trainer:
    __train_loss_key = 'train_loss'
    __test_loss_key = 'test_loss'

    def __init__(
            self,
            criterion: Module,
            optimizer: Optimizer,
            model: Module,
            device: str,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int
            ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.step_history = dict()
        self.history = dict()

    def fit(self):
        """The main training loop that train the model on the training
        data then test it on the test set and then log the results
        """
        for _ in range(self.epochs):
            self.train()
            self.test()
            self.print_results()

    def set_train_mode(self) -> None:
        """Set the models on the training mood
        """
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        """Set the models on the testing mood
        """
        self.model = self.model.eval()

    def print_results(self):
        """Prints the results after each epoch
        """
        result = ''
        for key, value in self.history.items():
            result += f'{key}: {str(value[-1])}, '
        print(result[:-2])

    def test(self):
        """Iterate over the whole test data and test the models
        for a single epoch
        """
        # TODO

    @save_checkpoint
    def train(self):
        """Iterates over the whole training data and train the models
        for a single epoch
        """
        total_loss = 0
        self.set_train_mode()
        for x, (y, target_lengths) in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            result = self.model(x)
            input_lengths = torch.full(
                size=(y.shape[0],),
                fill_value=y.shape[1],
                dtype=torch.long
                )
            loss = self.criterion(
                result,
                y,
                input_lengths,
                target_lengths
            )
            loss.backward()
            # TODO: adding optimizer schedular step
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        if self.__train_loss_key in self.history:
            self.history[self.__train_loss_key].append(total_loss)
        else:
            self.history[self.__train_loss_key] = [total_loss]
