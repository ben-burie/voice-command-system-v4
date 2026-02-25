import logging
from pathlib import Path

import torch

from model.classifier import WhisperCommandClassifier

logger = logging.getLogger(__name__)

def save_checkpoint(path: str, model: WhisperCommandClassifier, label_to_idx: dict, idx_to_label: dict, 
                    whisper_model_name: str, freeze_encoder: bool, val_acc: float, epoch: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "whisper_model_name": whisper_model_name,
            "freeze_encoder": freeze_encoder,
            "val_acc": val_acc,
            "epoch": epoch,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]
    whisper_model_name = ckpt["whisper_model_name"]
    freeze_encoder = ckpt.get("freeze_encoder", True)

    model = WhisperCommandClassifier(whisper_model_name, len(label_to_idx), freeze_encoder)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.1f}%)"
    )
    return model, label_to_idx, idx_to_label, whisper_model_name, freeze_encoder