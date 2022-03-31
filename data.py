from torchaudio.transforms import (
    Resample,
    TimeMasking,
    MelSpectrogram,
    FrequencyMasking
)
from utils import (
    IPipeline,
    load_audio
    )
from pathlib import Path
from typing import Union
from torch import Tensor


class AudioPipeline(IPipeline):
    """Loads the audio and pass it through different transformation layers
    """
    def __init__(
            self,
            sampling_rate: int,
            n_mel_channels: int,
            win_length: int,
            hop_length: int,
            n_time_masks: int,
            ps: float,
            max_freq_mask: int
            ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_time_masks = n_time_masks
        self.ps = ps
        self.max_freq_mask = max_freq_mask

    def run(
            self,
            audio_path: Union[str, Path],
            *args,
            **kwargs
            ) -> Tensor:
        x, sr = load_audio(audio_path)
        x = self._get_resampler(sr)(x)
        x = self._get_mel_spec_transformer()(x)
        x = self._mask(x)
        x = x.permute(0, 2, 1)
        return x

    def _get_resampler(self, sr: int):
        return Resample(sr, self.sampling_rate)

    def _get_mel_spec_transformer(self):
        return MelSpectrogram(
            self.sampling_rate,
            n_mels=self.n_mel_channels
            )

    @property
    def freq_mask(self):
        return FrequencyMasking(self.F)

    def _get_time_mask(self, max_length: int):
        return TimeMasking(
            int(max_length * self.ps),
            p=self.ps
        )

    def _mask(self, x: Tensor) -> Tensor:
        max_length = x.shape[-1]
        time_mask = self._get_time_mask(max_length)
        x = self.freq_mask(x)
        for _ in range(self.n_time_masks):
            x = time_mask(x)
        return x


class TextPipeline(IPipeline):
    """pass the text through different transformation layers
    """
    def __init__(self) -> None:
        super().__init__()

    def run(
            self,
            text: str
            ) -> str:
        text = text.lower()
        text = text.strip()
        return text
