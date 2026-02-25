import logging

import torch
import torch.nn as nn

from model.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, device, epochs: int, lr: float, checkpoint_path: str, 
                label_to_idx: dict, idx_to_label: dict, whisper_model_name: str, freeze_encoder: bool) -> None:
    """Run train/val loop for `epochs`, saving best checkpoint by val accuracy."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for mels, label_idxs, n_frames in train_loader:
            mels = mels.to(device)
            label_idxs = label_idxs.to(device)
            n_frames = n_frames.to(device)

            optimizer.zero_grad()
            logits = model(mels, n_frames)
            loss = criterion(logits, label_idxs)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_correct += (logits.argmax(1) == label_idxs).sum().item()
            t_total += label_idxs.size(0)

        # --- Validate ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for mels, label_idxs, n_frames in val_loader:
                mels = mels.to(device)
                label_idxs = label_idxs.to(device)
                n_frames = n_frames.to(device)

                logits = model(mels, n_frames)
                loss = criterion(logits, label_idxs)
                v_loss += loss.item()
                v_correct += (logits.argmax(1) == label_idxs).sum().item()
                v_total += label_idxs.size(0)

        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        logger.info(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train {t_loss / len(train_loader):.4f} / {t_acc:.1f}% | "
            f"Val {v_loss / len(val_loader):.4f} / {v_acc:.1f}%"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(
                checkpoint_path, model, label_to_idx, idx_to_label,
                whisper_model_name, freeze_encoder, v_acc, epoch + 1,
            )
            logger.info(f"  → Best checkpoint saved (val_acc={v_acc:.1f}%)")

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.1f}%")