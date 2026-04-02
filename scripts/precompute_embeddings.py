"""
Pre-compute Whisper encoder embeddings for a test directory and save to a .pt cache.

The cache stores pooled encoder embeddings (one vector per file) so that
evaluate_faster.py can skip the expensive encoder forward pass entirely.

Embeddings depend only on the Whisper model variant (tiny/base/small/medium/turbo),
not on any specific checkpoint. The same cache works for all frozen-encoder checkpoints
that share the same --whisper-model size.

Usage:
    python scripts/precompute_embeddings.py --test-dir test_data_3cmd
    python scripts/precompute_embeddings.py --test-dir test_data_3cmd --whisper-model turbo
    python scripts/precompute_embeddings.py --test-dir test_data_3cmd --output embeddings/3cmd.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import whisper

from audio.preprocessing import preprocess_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _scan_test_dir(test_dir: Path) -> dict[str, list[Path]]:
    result = {}
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        wavs = sorted(subdir.glob("*.wav"))
        if wavs:
            result[subdir.name] = wavs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute Whisper encoder embeddings for a test directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-dir", required=True, help="Root test directory (one subfolder per label)")
    parser.add_argument(
        "--whisper-model", default="turbo",
        choices=["tiny", "base", "small", "medium", "turbo"],
        help="Whisper model variant to use as encoder",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .pt file path (default: <test_dir>_embeddings.pt next to test dir)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoder forward-pass batch size")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    output_path = Path(args.output) if args.output else test_dir.parent / f"{test_dir.name}_embeddings.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading Whisper encoder (%s)...", args.whisper_model)
    t0 = time.time()
    base = whisper.load_model(args.whisper_model, device=device)
    encoder = base.encoder
    encoder.eval()
    n_mels = base.dims.n_mels
    logger.info("Encoder loaded in %.1fs  (n_mels=%d)", time.time() - t0, n_mels)

    test_data = _scan_test_dir(test_dir)
    if not test_data:
        logger.error("No .wav files found under %s", test_dir)
        sys.exit(1)

    total_wav_count = sum(len(v) for v in test_data.values())
    logger.info("Found %d files across %d classes", total_wav_count, len(test_data))

    # --- Step 1: preprocess all wav files to mel spectrograms ---
    logger.info("Preprocessing %d audio files...", total_wav_count)
    t0 = time.time()
    all_labels: list[str] = []
    all_filenames: list[str] = []
    all_mels: list[torch.Tensor] = []
    all_n_frames: list[int] = []

    processed = 0
    for label, wav_paths in sorted(test_data.items()):
        for wav_path in wav_paths:
            try:
                mel, n_frames = preprocess_audio(str(wav_path), n_mels=n_mels)
            except Exception as e:
                logger.warning("Skipping %s — preprocessing failed: %s", wav_path.name, e)
                continue
            all_labels.append(label)
            all_filenames.append(wav_path.name)
            all_mels.append(mel)
            all_n_frames.append(n_frames)
            processed += 1
            if processed % 50 == 0:
                logger.info("  Preprocessed %d / %d...", processed, total_wav_count)

    logger.info("Preprocessing done: %d files in %.1fs", processed, time.time() - t0)

    # --- Step 2: batched encoder forward pass to get pooled embeddings ---
    logger.info("Running encoder on %d files (batch_size=%d)...", processed, args.batch_size)
    t0 = time.time()
    all_embeddings: list[torch.Tensor] = []
    n_batches = (processed + args.batch_size - 1) // args.batch_size

    for batch_num, start in enumerate(range(0, processed, args.batch_size), start=1):
        end = start + args.batch_size
        batch_mels = torch.stack(all_mels[start:end]).to(device)
        batch_frames = torch.tensor(all_n_frames[start:end], device=device)
        logger.info("  Batch %d / %d (%d files)...", batch_num, n_batches, len(batch_mels))

        with torch.no_grad():
            features = encoder(batch_mels)  # [B, T, hidden_dim]
            B, T, H = features.shape
            enc_frames = (batch_frames // 2).clamp(min=1)
            mask = (torch.arange(T, device=device).unsqueeze(0) < enc_frames.unsqueeze(1))
            mask = mask.unsqueeze(-1).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, hidden_dim]

        all_embeddings.append(pooled.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [N, hidden_dim]
    logger.info("Encoder done in %.1fs  — embedding shape: %s", time.time() - t0, list(embeddings_tensor.shape))

    # --- Step 3: save cache ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "embeddings": embeddings_tensor,
        "labels": all_labels,
        "filenames": all_filenames,
        "whisper_model_name": args.whisper_model,
        "test_dir": str(test_dir),
    }
    torch.save(cache, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Saved %d embeddings → %s  (%.1f MB)", processed, output_path, size_mb)


if __name__ == "__main__":
    main()
