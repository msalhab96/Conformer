import math
import pandas as pd
import torch
from tokenizer import ITokenizer
from utils import IPipeline
from pathlib import Path
from typing import List, Union
from torch import Tensor


class BaseData:
    def __init__(
            self,
            text_pipeline: IPipeline,
            audio_pipeline: IPipeline,
            tokenizer: ITokenizer,
            sampling_rate: int,
            hop_length: int,
            fields_sep: str,
            csv_file_keys: object
            ) -> None:
        self.text_pipeline = text_pipeline
        self.audio_pipeline = audio_pipeline
        self.tokenizer = tokenizer
        self.max_len = 0
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.sep = fields_sep
        self.csv_file_keys = csv_file_keys

    def _get_padded_aud(
            self,
            aud_path: Union[str, Path],
            max_duration: int,
            ) -> Tensor:
        max_len = 1 + math.ceil(
            max_duration * self.sampling_rate / self.hop_length
            )
        # Updates teh max_len to be used for text padding
        self.max_len = max_len
        aud = self.audio_pipeline.run(aud_path)
        assert aud.shape[0] == 1, f'expected audio of 1 channels got \
            {aud_path} with {aud.shape[0]} channels'
        return self.pad_mels(aud, max_len)

    def _get_padded_tokens(self, text: str) -> Tensor:
        text = self.text_pipeline.run(text)
        tokens = self.tokenizer.tokens2ids(text)
        length = len(tokens)
        tokens = self.pad_tokens(tokens)
        return torch.LongTensor(tokens), length

    def prepocess_lines(self, data: str) -> List[str]:
        return [
            item.split(self.sep)
            for item in data
        ]

    def pad_mels(self, mels: Tensor, max_len: int) -> Tensor:
        n = max_len - mels.shape[1]
        zeros = torch.zeros(size=(1, n, mels.shape[-1]))
        return torch.cat([zeros, mels], dim=1)

    def pad_tokens(self, tokens: list) -> Tensor:
        length = self.max_len - len(tokens)
        return tokens + [self.tokenizer.special_tokens.pad_id] * length


class DataLoader(BaseData):
    def __init__(
            self,
            file_path: Union[str, Path],
            text_pipeline: IPipeline,
            audio_pipeline: IPipeline,
            tokenizer: ITokenizer,
            batch_size: int,
            sampling_rate: int,
            hop_length: int,
            fields_sep: str,
            csv_file_keys: object
            ) -> None:
        super().__init__(
                text_pipeline,
                audio_pipeline,
                tokenizer,
                sampling_rate,
                hop_length,
                fields_sep,
                csv_file_keys
                )
        self.batch_size = batch_size
        self.df = pd.read_csv(file_path)
        self.num_examples = len(self.df)
        self.idx = 0

    def __len__(self):
        length = self.num_examples // self.batch_size
        mod = self.num_examples % self.batch_size
        return length + 1 if mod > 0 else length

    def get_max_duration(self, start_idx: int, end_idx: int) -> float:
        return self.df[
            self.csv_file_keys.duration
            ].iloc[start_idx: end_idx].max()

    def get_audios(self, start_idx: int, end_idx: int) -> Tensor:
        max_duration = self.get_max_duration(start_idx, end_idx)
        result = list(map(
            self._get_padded_aud,
            self.df[self.csv_file_keys.path].iloc[start_idx: end_idx],
            [max_duration] * (end_idx - start_idx)
            ))
        result = torch.stack(result, dim=1)
        return torch.squeeze(result)

    def get_texts(self, start_idx: int, end_idx: int) -> Tensor:
        args = self.df[self.csv_file_keys.text].iloc[start_idx: end_idx]
        result = list(map(self._get_padded_tokens, args))
        length = list(map(lambda x: x[1], result))
        result = list(map(lambda x: x[0], result))
        result = torch.stack(result, dim=0)
        return result, torch.LongTensor(length)

    def __iter__(self):
        self.idx = 0
        while True:
            start = self.idx * self.batch_size
            end = (self.idx + 1) * self.batch_size
            end = min(end, self.num_examples)
            if start > self.num_examples or start == end:
                break
            self.idx += 1
            yield (
                self.get_audios(start, end),
                self.get_texts(start, end)
                )
