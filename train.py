import os
from pathlib import Path
import torch
from hprams import (
    get_audpipe_params,
    get_model_params,
    get_optim_params,
    hprams
    )
from functools import wraps
from torch.nn import Module
from data import DataLoader
from torch.optim import Optimizer
from typing import Callable, Union
from tqdm import tqdm
from model import Model

from optim import AdamWarmup
from pipelines import get_pipelines
from tokenizer import CharTokenizer, ITokenizer
from utils import IPipeline

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
        total_loss = 0
        self.set_test_mode()
        for x, (y, target_lengths) in tqdm(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            result = self.model(x)
            result = result.log_softmax(axis=-1)
            result = result.permute(1, 0, 2)
            y = y[..., :result.shape[0]]
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
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        if self.__test_loss_key in self.history:
            self.history[self.__test_loss_key].append(total_loss)
        else:
            self.history[self.__test_loss_key] = [total_loss]

    @save_checkpoint
    def train(self):
        """Iterates over the whole training data and train the models
        for a single epoch
        """
        torch.autograd.set_detect_anomaly(True)
        total_loss = 0
        self.set_train_mode()
        for x, (y, target_lengths) in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            result = self.model(x)
            result = result.log_softmax(axis=-1)
            result = result.permute(1, 0, 2)
            y = y[..., :result.shape[0]]
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
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        if self.__train_loss_key in self.history:
            self.history[self.__train_loss_key].append(total_loss)
        else:
            self.history[self.__train_loss_key] = [total_loss]


def get_criterion(blank_id: int) -> Module:
    return torch.nn.CTCLoss(blank_id)


def get_optimizer(model: Module, params: dict) -> object:
    return AdamWarmup(
        model.parameters(),
        **params
        )


def load_model(
        model_params: dict,
        checkpoint_path=None
        ) -> Module:
    model = Model(**model_params)
    if checkpoint_path is not None:
        model.load_state_dict(
            torch.load(checkpoint_path)[MODEL_KEY]
            )
    return model


def get_data_loader(
        file_path: Union[str, Path],
        tokenizer: ITokenizer,
        text_pipeline: IPipeline,
        audio_pipeline: IPipeline,
        ):
    return DataLoader(
        file_path,
        text_pipeline,
        audio_pipeline,
        tokenizer,
        hprams.training.batch_size,
        hprams.data.sampling_rate,
        hprams.data.hop_length,
        hprams.data.files_sep,
        hprams.data.csv_file_keys,
    )


def get_tokenizer():
    tokenizer = CharTokenizer()
    if hprams.tokenizer.tokenizer_file is not None:
        tokenizer = tokenizer.load_tokenizer(
            hprams.tokenizer.tokenizer_file
            )
    tokenizer = tokenizer.add_blank_token().add_pad_token()
    with open(hprams.tokenizer.vocab_path, 'r') as f:
        vocab = f.read().split('\n')
    tokenizer.set_tokenizer(vocab)
    tokenizer.save_tokenizer('tokenizer.json')
    return tokenizer


def get_train_test_loaders(
        tokenizer: ITokenizer,
        audio_pipeline: IPipeline,
        text_pipeline: IPipeline
        ) -> tuple:
    return (
        get_data_loader(
            hprams.data.training_file,
            tokenizer,
            text_pipeline,
            audio_pipeline
            ),
        get_data_loader(
            hprams.data.testing_file,
            tokenizer,
            text_pipeline,
            audio_pipeline
            )
    )


def get_trainer() -> Trainer:
    device = hprams.device
    tokenizer = get_tokenizer()
    blank_id = tokenizer.special_tokens.blank_id
    vocab_size = tokenizer.vocab_size
    text_pipeline, audio_pipeline = get_pipelines(
        get_audpipe_params()
    )
    model = load_model(
        get_model_params(vocab_size),
        checkpoint_path=hprams.checkpoint
        ).to(device)
    train_loader, test_loader = get_train_test_loaders(
        tokenizer,
        audio_pipeline,
        text_pipeline
    )
    return Trainer(
        criterion=get_criterion(blank_id),
        optimizer=get_optimizer(
            model, get_optim_params()
            ),
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=hprams.training.epochs
    )


if __name__ == '__main__':
    trainer = get_trainer()
    trainer.fit()
