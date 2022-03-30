from abc import ABC, abstractmethod
from typing import  Union
from os import PathLike
import json


class IDataLoader(ABC):

    @abstractmethod
    def load(self):
        pass

class JSONLoader(IDataLoader):
    def __init__(self, file_path: Union[str, PathLike]) -> None:
        super().__init__()
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data

class TextLoader(IDataLoader):
    def __init__(self, file_path: Union[str, PathLike]) -> None:
        super().__init__()
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as f:
            data = f.read()
        return data
