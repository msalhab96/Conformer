from abc import ABC, abstractmethod
from typing import Union, Tuple
from pathlib import Path
from torch import Tensor
import torchaudio
import json


def save_json(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


class IPipeline(ABC):
    @abstractmethod
    def run():
        """Used to run all the callables functions sequantially
        """
        pass


def load_audio(file_path: Union[str, Path]) -> Tuple[Tensor, int]:
    x, sr = torchaudio.load(file_path, normalize=True)
    return x, sr