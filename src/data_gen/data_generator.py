"""
BarkAI synthetic audio generation for voice command training data.

Public API:
    load_models()                  — call once before any generation
    generate_audio_for_command(...)  — generate .wav files for one command label
"""
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.io.wavfile import write as write_wav

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

from bark import SAMPLE_RATE, generate_audio, preload_models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via generate_audio_for_command kwargs)
# ---------------------------------------------------------------------------
SPEAKERS = [f"v2/en_speaker_{i}" for i in range(10)]
BASE_SAMPLES_PER_SPEAKER = 2
NUM_IO_THREADS = 4

AUGMENTATION_POLICY = {
    "noise":       {"enabled": True,      "snr_db":   [30, 20, 10]},
    "tempo":       {"enabled": _LIBROSA,  "factors":  [0.85, 0.9, 0.95, 1.05, 1.1]},
    "pitch":       {"enabled": _LIBROSA,  "steps":    [-2, -1, 1, 2]},
    "compression": {"enabled": True,      "ratios":   [4, 8]},
    "eq":          {"enabled": True,      "bands":    ["low_cut", "mid_boost", "high_cut"]},
    "reverb":      {"enabled": True,      "strength": [0.3, 0.6]},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models() -> None:
    """Preload Bark AI models into memory. Call once before generation loops."""
    logger.info("Loading Bark AI models...")
    preload_models()
    logger.info("Bark models loaded.")


# ---------------------------------------------------------------------------
# Speech text variations
# ---------------------------------------------------------------------------
def _create_speech_variations(command: str) -> List[str]:
    words = command.split()
    variations = [
        command, command.lower(), command.upper(),
        f"Um, {command}", f"Uh, {command}", f"Hmm, {command}", f"Ah, {command}",
        f"{command}, please", f"{command} if you don't mind",
        f"Could you {command.lower()}",
        f"{command}!", f"Please {command}",
        f"{command}.", f"{command} now", f"{command} please",
    ]
    if len(words) > 2:
        mid = len(words) // 2
        variations.append(f"{' '.join(words[:mid])}... {' '.join(words[mid:])}")
        variations.append(f"{words[0]}... {' '.join(words[1:])}")

    seen: set = set()
    unique = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# ---------------------------------------------------------------------------
# Audio augmentations
# ---------------------------------------------------------------------------
def _apply_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    power = np.mean(audio ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    return audio + np.random.normal(0, np.sqrt(noise_power), len(audio))


def _apply_compression(audio: np.ndarray, ratio: float, threshold_db: float = -20) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20)
    out = np.copy(audio)
    mask = np.abs(audio) > threshold
    out[mask] = np.sign(audio[mask]) * threshold + (np.abs(audio[mask]) - threshold) / ratio
    return out


def _apply_eq(audio: np.ndarray, band: str) -> np.ndarray:
    nyquist = SAMPLE_RATE / 2
    if band == "low_cut":
        sos = signal.butter(4, 300 / nyquist, btype="high", output="sos")
    elif band == "mid_boost":
        sos = signal.butter(4, [800 / nyquist, 3000 / nyquist], btype="band", output="sos")
    elif band == "high_cut":
        sos = signal.butter(4, 7000 / nyquist, btype="low", output="sos")
    else:
        return audio
    return signal.sosfilt(sos, audio)


def _apply_reverb(audio: np.ndarray, strength: float) -> np.ndarray:
    base_delay = int(0.05 * SAMPLE_RATE)
    delays = [base_delay, int(base_delay * 1.5), int(base_delay * 2.0)]
    decays = [strength * 0.5, strength * 0.3, strength * 0.15]
    reverb = np.zeros_like(audio)
    for delay, decay in zip(delays, decays):
        delayed = np.zeros_like(audio)
        delayed[delay:] = audio[:-delay] * decay
        reverb += delayed
    return audio + reverb


def _apply_augmentations(audio: np.ndarray, policy: dict) -> List[np.ndarray]:
    versions = [audio]

    if policy["noise"]["enabled"]:
        for snr in policy["noise"]["snr_db"]:
            try:
                versions.append(_apply_noise(audio, snr))
            except Exception as e:
                logger.warning("Noise aug failed (snr=%s): %s", snr, e)

    if _LIBROSA and policy["tempo"]["enabled"]:
        for factor in policy["tempo"]["factors"]:
            try:
                versions.append(librosa.effects.time_stretch(audio, rate=factor))
            except Exception as e:
                logger.warning("Tempo aug failed (factor=%s): %s", factor, e)

    if _LIBROSA and policy["pitch"]["enabled"]:
        for steps in policy["pitch"]["steps"]:
            try:
                versions.append(librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps))
            except Exception as e:
                logger.warning("Pitch aug failed (steps=%s): %s", steps, e)

    if policy["compression"]["enabled"]:
        for ratio in policy["compression"]["ratios"]:
            try:
                versions.append(_apply_compression(audio, ratio))
            except Exception as e:
                logger.warning("Compression aug failed (ratio=%s): %s", ratio, e)

    if policy["eq"]["enabled"]:
        for band in policy["eq"]["bands"]:
            try:
                versions.append(_apply_eq(audio, band))
            except Exception as e:
                logger.warning("EQ aug failed (band=%s): %s", band, e)

    if policy["reverb"]["enabled"]:
        for strength in policy["reverb"]["strength"]:
            try:
                versions.append(_apply_reverb(audio, strength))
            except Exception as e:
                logger.warning("Reverb aug failed (strength=%s): %s", strength, e)

    return versions


def _normalize(audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio * (10 ** (target_lufs / 20) / rms)
    return np.tanh(audio)


# ---------------------------------------------------------------------------
# Parallel I/O
# ---------------------------------------------------------------------------
def _save_batch(batch: List[Tuple[np.ndarray, Path]]) -> None:
    def _save(audio: np.ndarray, path: Path) -> None:
        write_wav(str(path), SAMPLE_RATE, audio)

    with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as pool:
        futures = [pool.submit(_save, audio, path) for audio, path in batch]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Failed to save audio file: %s", e)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_audio_for_command(
    label: str,
    display_name: str,
    n_samples: int,
    out_dir: Path,
    speakers: List[str] = SPEAKERS,
    base_samples_per_speaker: int = BASE_SAMPLES_PER_SPEAKER,
    policy: dict = None,
    seed: int = 42,
) -> int:
    """
    Generate up to n_samples .wav files for a single command label.

    Files are written to out_dir/<label>/ and named
    <label>_speaker<NN>_aug<NNNN>.wav

    Returns the number of files actually written.
    """
    if policy is None:
        policy = AUGMENTATION_POLICY

    random.seed(seed)
    np.random.seed(seed)

    out_path = out_dir / label
    out_path.mkdir(parents=True, exist_ok=True)

    variations = _create_speech_variations(display_name)
    logger.info("Generating '%s': %d text variations, target %d samples", label, len(variations), n_samples)

    count = 0
    for speaker_idx, speaker in enumerate(speakers):
        for base_idx in range(base_samples_per_speaker):
            if count >= n_samples:
                break

            var_idx = (speaker_idx * base_samples_per_speaker + base_idx) % len(variations)
            text = variations[var_idx]

            try:
                raw = generate_audio(text, history_prompt=speaker)
                raw = _normalize(raw)
                augmented = _apply_augmentations(raw, policy)
            except Exception as e:
                logger.warning("Bark generation failed (speaker=%s, text=%r): %s", speaker, text, e)
                continue

            batch = []
            for aug in augmented:
                if count >= n_samples:
                    break
                audio = np.array(aug, dtype=np.float32)
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio * (0.95 / max_val)
                filename = f"{label}_speaker{speaker_idx:02d}_aug{count:04d}.wav"
                batch.append((audio, out_path / filename))
                count += 1

            _save_batch(batch)

        if count >= n_samples:
            break

    logger.info("'%s': wrote %d files to %s", label, count, out_path)
    return count
