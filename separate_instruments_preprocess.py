import pathlib

import librosa as li
import numpy as np
import yaml
from tqdm import tqdm

from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config


def get_files(data_location: str, extension: str, **_) -> list[pathlib.Path]:
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def split_files(files, val_ratio=0.1, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))
    n = len(files)
    n_test = max(1, int(test_ratio * n))
    n_val  = max(1, int(val_ratio * n))
    test  = files[:n_test]
    val   = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
    return train, val, test


def save_subdataset(signals, pitches, loudness, out_path):
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "signals.npy", signals)
    np.save(out_path / "pitches.npy", pitches)
    np.save(out_path / "loudness.npy", loudness)


def preprocess(
    f, sampling_rate, block_size, signal_length, oneshot, **_):
    '''Preprocess a single audio file.
    Args:
        f: Path to the audio file.
        sampling_rate: Sampling rate to load the audio file.
        block_size: Block size for pitch and loudness extraction.
        signal_length: Length of the output signal segments.
        oneshot: If True, only process the first segment of the audio file.'''
    
    x, _ = li.load(f, sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


def process_files(files, config):
    if not files:
        raise ValueError(f"No files to process — dataset may be too small to split.")
    
    signals: list[np.ndarray] = []
    pitches: list[np.ndarray] = []
    loudness: list[np.ndarray] = []

    progress_bar = tqdm(files)
    for f in progress_bar:
        progress_bar.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitches.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitches = np.concatenate(pitches, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    return (signals, pitches, loudness)


def main():
    class args(Config):
        CONFIG = "config.yaml"
        SPLIT_DATASET = True

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    files = get_files(**config["data"])

    instruments = config["data"]["instruments"]

    for instrument in instruments:

        files_instrument = [f for f in files if f.stem.split("_")[2] == instrument]
        out_path = pathlib.Path(config["preprocess"]["out_dir"]) / instrument

        if args.SPLIT_DATASET:
            train_files, val_files, test_files = split_files(files_instrument)
            for subset, subset_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
                signals, pitches, loudness = process_files(subset_files, config)
                save_subdataset(signals, pitches, loudness, out_path / subset)
        else:
            train_files, val_files, test_files = files_instrument, [], []
            signals, pitches, loudness = process_files(train_files, config)
            save_subdataset(signals, pitches, loudness, out_path / "train")


if __name__ == "__main__":
    main()