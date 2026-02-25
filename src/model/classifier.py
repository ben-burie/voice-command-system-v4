import logging

import torch
import torch.nn as nn
import whisper

logger = logging.getLogger(__name__)


class WhisperCommandClassifier(nn.Module):
    """Whisper encoder (frozen by default) → mean-pool over speech frames → linear classifier."""

    def __init__(self, whisper_model_name: str, num_classes: int, freeze_encoder: bool = True):
        super().__init__()
        base = whisper.load_model(whisper_model_name)
        self.encoder = base.encoder
        self.n_mels = base.dims.n_mels
        self._hidden_dim = base.dims.n_audio_state

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("Encoder frozen — training classification head only.")
        else:
            logger.info("Full fine-tune — encoder + classification head.")

        self.classifier = nn.Linear(self._hidden_dim, num_classes)

    def forward(self, mel: torch.Tensor, n_frames: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(mel)  # [B, T, hidden_dim]

        if n_frames is not None:
            # Encoder's stride-2 conv halves 3000 mel frames → 1500 encoder frames
            B, T, H = features.shape
            enc_frames = (n_frames // 2).clamp(min=1)
            mask = (torch.arange(T, device=features.device).unsqueeze(0) < enc_frames.unsqueeze(1))
            mask = mask.unsqueeze(-1).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = features.mean(dim=1)

        return self.classifier(pooled)  # [B, num_classes]
