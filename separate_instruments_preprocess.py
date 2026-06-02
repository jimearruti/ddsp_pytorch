import pathlib

import librosa as li
import numpy as np
import yaml
from tqdm import tqdm

from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config


def get_files(data_location: str, extension: str, **_) -> list[pathlib.Path]:
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))

def preprocess(
    f: str | pathlib.Path,
    sampling_rate: int,
    block_size: int,
    signal_length: int,
    oneshot: bool,
    **_,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def main():

    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    files = get_files(**config["data"])

    instruments = config["data"]["instruments"]

    for instrument in instruments:

        files_instrument = [f for f in files if f.stem.split("_")[2] == instrument]
        progress_bar = tqdm(files_instrument)

        signals: list[np.ndarray] = []
        pitches: list[np.ndarray] = []
        loudness: list[np.ndarray] = []

        for f in progress_bar:
            progress_bar.set_description(str(f))
            x, p, l = preprocess(f, **config["preprocess"])
            signals.append(x)
            pitches.append(p)
            loudness.append(l)

        signals = np.concatenate(signals, 0).astype(np.float32)
        pitches = np.concatenate(pitches, 0).astype(np.float32)
        loudness = np.concatenate(loudness, 0).astype(np.float32)

        out_path = pathlib.Path(config["preprocess"]["out_dir"]) / instrument
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "signals.npy", signals)
        np.save(out_path / "pitches.npy", pitches)
        np.save(out_path / "loudness.npy", loudness)


if __name__ == "__main__":
    main()