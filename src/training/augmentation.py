import logging

import torch
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class SpecAugment:
    """SpecAugment: randomly mask time frames and frequency bins on a mel spectrogram."""

    def __init__(self, time_mask_param: int = 50, freq_mask_param: int = 10):
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > 0.5:
            mel = self.time_masking(mel)
        if torch.rand(1) > 0.5:
            mel = self.freq_masking(mel)
        return mel