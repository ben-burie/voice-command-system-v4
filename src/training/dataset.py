import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from audio.preprocessing import preprocess_audio
from training.augmentation import SpecAugment

logger = logging.getLogger(__name__)


class CommandDataset(Dataset):
    def __init__(self, file_paths, labels, label_to_idx, n_mels=80, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.n_mels = n_mels
        self.augment = SpecAugment() if augment else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel, n_frames = preprocess_audio(self.file_paths[idx], self.n_mels)
        if self.augment:
            mel = self.augment(mel)
        return mel, self.label_to_idx[self.labels[idx]], n_frames


def load_data_from_dir(data_dir: str) -> dict[str, list[str]]:
    """Scan data_dir for subdirectories; each subdirectory is a command class with .wav files."""
    data = {}
    for cmd_dir in sorted(Path(data_dir).iterdir()):
        if cmd_dir.is_dir():
            files = [str(f) for f in cmd_dir.iterdir() if f.suffix.lower() == ".wav"]
            if files:
                data[cmd_dir.name] = files
                logger.info(f"  {cmd_dir.name}: {len(files)} files")
    return data


def build_dataloaders(
    data_dict: dict, label_to_idx: dict, n_mels: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """80/20 stratified split → (train_loader, val_loader). Train set gets SpecAugment."""
    file_paths, labels = [], []
    for label, paths in data_dict.items():
        file_paths.extend(paths)
        labels.extend([label] * len(paths))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = CommandDataset(train_paths, train_labels, label_to_idx, n_mels, augment=True)
    val_ds = CommandDataset(val_paths, val_labels, label_to_idx, n_mels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader