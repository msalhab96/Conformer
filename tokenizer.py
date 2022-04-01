from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Tuple,
    Union
    )
from os import PathLike
from data_loaders import JSONLoader
from utils import save_json
from functools import wraps

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
BLANK = '<BLANK>'


def check_token(token: str) -> Callable:
    """To check if a token exists or not

    Args:
        token ([type]): the token to be checked
    """
    def decorator(func):
        @wraps(func)
        def wrapper(obj, token=token):
            if token in obj._token_to_id:
                return obj._token_to_id[token]
            return func(obj, token)
        return wrapper
    return decorator


@dataclass
class SpecialTokens:
    _pad: Tuple[str, int] = (None, None)
    _blank: Tuple[str, int] = (None, None)
    _sos: Tuple[str, int] = (None, None)
    _eos: Tuple[str, int] = (None, None)

    @property
    def pad_id(self):
        return self._pad[1]

    @property
    def pad_token(self):
        return self._pad[0]

    @property
    def blank_id(self):
        return self._blank[1]

    @property
    def blank_token(self):
        return self._blank[0]

    @property
    def sos_id(self):
        return self._sos[1]

    @property
    def sos_token(self):
        return self._sos[0]

    @property
    def eos_id(self):
        return self._eos[1]

    @property
    def eos_token(self):
        return self._eos[0]

    @property
    def mask_id(self):
        return self._mask[1]

    @property
    def mask_token(self):
        return self._mask[0]


class ITokenizer(ABC):

    @abstractmethod
    def ids2tokens(self):
        pass

    @abstractmethod
    def tokens2ids(self):
        pass

    @abstractmethod
    def set_tokenizer(self):
        pass

    @abstractmethod
    def save_tokenizer(self):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def add_token(self):
        pass

    @abstractmethod
    def preprocess_tokens(self):
        pass

    @abstractmethod
    def batch_tokenizer(self):
        pass

    @abstractproperty
    def vocab_size(self):
        pass

    @abstractmethod
    def get_tokens(self):
        pass


class BaseTokenizer(ITokenizer):
    _pad_key = 'pad'
    _sos_key = 'sos'
    _eos_key = 'eos'
    _blank_key = 'blank'
    _token_to_id_key = 'token_to_id'
    _special_tokens_key = 'special_tokens'

    def __init__(self) -> None:
        super().__init__()
        self._token_to_id = dict()
        self._id_to_token = dict()
        self.special_tokens = SpecialTokens()

    @property
    def vocab_size(self):
        return len(self._token_to_id)

    def add_token(self, token: str):
        token_id = self.vocab_size
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        return token_id

    @check_token(PAD)
    def add_pad_token(self, token=PAD) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._pad = (token, token_id)
        return self

    @check_token(BLANK)
    def add_blank_token(self, token=BLANK) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._blank = (token, token_id)
        return self

    @check_token(SOS)
    def add_sos_token(self, token=SOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._sos = (token, token_id)
        return self

    @check_token(EOS)
    def add_eos_token(self, token=EOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._eos = (token, token_id)
        return self

    def _reset_id_to_token(self) -> None:
        self._id_to_token = dict(zip(
            self._token_to_id.values(),
            self._token_to_id.keys()
            ))

    def __set_special_tokens_dict(self, data: dict) -> None:
        if self._pad_key in data:
            self.special_tokens._pad = tuple(data[self._pad_key])
        if self._blank_key in data:
            self.special_tokens._blank = tuple(data[self._blank_key])
        if self._sos_key in data:
            self.special_tokens._sos = tuple(data[self._sos_key])
        if self._eos_key in data:
            self.special_tokens._eos = tuple(data[self._eos_key])

    def __get_special_tokens_dict(self) -> dict:
        data = {}
        if self.special_tokens.pad_id is not None:
            data[self._pad_key] = list(self.special_tokens._pad)
        if self.special_tokens.blank_id is not None:
            data[self._blank_key] = list(self.special_tokens._blank)
        if self.special_tokens.sos_id is not None:
            data[self._sos_key] = list(self.special_tokens._sos)
        if self.special_tokens.eos_id is not None:
            data[self._eos_key] = list(self.special_tokens._eos)
        return data

    def load_tokenizer(
            self,
            tokenizer_path: Union[str, PathLike],
            *args,
            **kwargs
            ) -> ITokenizer:
        data = JSONLoader(tokenizer_path).load()
        self._token_to_id = data[self._token_to_id_key]
        self.__set_special_tokens_dict(data[self._special_tokens_key])
        self._reset_id_to_token()
        return self

    def set_tokenizer(self, data: List[str], *args, **kwargs) -> ITokenizer:
        all_tokens = self.get_tokens(data)
        _ = list(map(self.add_token, all_tokens))
        self._reset_id_to_token()
        return self

    def save_tokenizer(
            self,
            save_path: Union[str, PathLike],
            *args,
            **kwargs
            ) -> None:
        data = {
            self._token_to_id_key: self._token_to_id,
            self._special_tokens_key: self.__get_special_tokens_dict()
        }
        save_json(save_path, data)

    def ids2tokens(self, ids: List[str]) -> List[str]:
        return list(map(lambda x: self._id_to_token[x], ids))

    def tokens2ids(self, sentence: str) -> List[int]:
        sentence = self.preprocess_tokens(sentence)
        return list(map(
            lambda x: self._token_to_id.get(x, self.special_tokens.pad_id),
            sentence)
            )

    def batch_tokenizer(self, data: List[str]) -> list:
        return list(map(self.tokens2ids, data))

    def batch_detokenizer(self, data: List[int]) -> list:
        return list(map(self.ids2tokens, data))


class CharTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()

    def get_tokens(self, data: List[str]):
        return set(''.join(data))

    def preprocess_tokens(self, sentence: str) -> List[str]:
        return list(sentence)
