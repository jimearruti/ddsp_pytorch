import datetime
import pathlib
import numpy as np
import yaml
from effortless_config import Config

def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()

    with open(args.CONFIG, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    out_dir = pathlib.Path(config["preprocess"]["out_dir"])
    instruments = config["data"]["instruments"]
    sampling_rate = config["preprocess"]["sampling_rate"]

    duration_summary = {}

    print("Analyzing preprocessed data...")
    
    for instrument in instruments:
        signal_file = out_dir / instrument / "signals.npy"
        
        if not signal_file.exists():
            print(f"[WARNING] No signals.npy found for {instrument} at {signal_file}")
            continue

        signals_mmap = np.load(signal_file, mmap_mode="r")
        total_samples = signals_mmap.size

        total_seconds = total_samples / sampling_rate
        duration_formatted = str(datetime.timedelta(seconds=int(total_seconds)))
        
        duration_summary[instrument] = duration_formatted

    print("\n" + "="*40)
    print(f"{'Instrument':<20} | {'Total Duration':<15}")
    print("="*40)
    for inst, dur in duration_summary.items():
        print(f"{inst:<20} | {dur:<15}")
    print("="*40)

if __name__ == "__main__":
    main()