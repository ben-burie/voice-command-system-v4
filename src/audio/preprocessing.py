import logging

import torch
import whisper

logger = logging.getLogger(__name__)

# Whisper uses 16kHz audio; hop length 160 → 3000 frames = 30 seconds
_WHISPER_HOP_LENGTH = 160

def preprocess_audio(path: str, n_mels: int = 80) -> tuple[torch.Tensor, int]:
    """Load a WAV file and return (log-mel spectrogram [n_mels, 3000], n_speech_frames).

    n_speech_frames is the number of mel frames corresponding to real audio
    before Whisper's 30-second zero-padding. Pass this to the model so it
    can pool over speech only, not silence.
    """
    audio = whisper.load_audio(str(path))
    n_frames = min(len(audio) // _WHISPER_HOP_LENGTH, 3000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    return mel, n_frames
