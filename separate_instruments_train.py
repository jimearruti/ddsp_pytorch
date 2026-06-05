import os
import pathlib
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import wandb
import yaml
from dotenv import load_dotenv
from effortless_config import Config
from tqdm import tqdm

from ddsp.core import mean_std_loudness, multiscale_fft, safe_log
from ddsp.model import DDSP
from ddsp.utils import get_scheduler
from preprocess import DatasetMultiInstrument


load_dotenv()
wandb.login(key=os.environ.get("WANDB_API_KEY"))


# training config
class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 100000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    INSTRUMENT = None


args.parse_args()

# model config
with open(args.CONFIG, "r") as config_file:
    config = yaml.safe_load(config_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

instruments = (
    [args.INSTRUMENT] 
    if args.INSTRUMENT 
    else config["data"]["instruments"]
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


for instrument in instruments:
    save_path = pathlib.Path(args.ROOT) / args.NAME / timestamp / instrument
    save_path.mkdir(parents=True, exist_ok=True)

    model = DDSP(**config["model"]).to(device)

    dataset = DatasetMultiInstrument(config["preprocess"]["out_dir"], instrument)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.BATCH,
        True,
        drop_last=True
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    run = wandb.init(
        project=args.NAME,
        name=instrument,
        config={**config, "instrument": instrument},
    )

    with open(save_path / "config.yaml", "w") as out_config:
        yaml.safe_dump(config, out_config)

    opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

    schedule = get_scheduler(
        len(dataloader),
        args.START_LR,
        args.STOP_LR,
        args.DECAY_OVER,
    )

    best_loss = float("inf")
    mean_loss = 0.0
    n_element = 0
    step = 0
    epochs = int(np.ceil(args.STEPS / len(dataloader)))
 
    for e in tqdm(range(epochs)):
        for s, p, l in dataloader:
            s = s.to(device)
            p = p.unsqueeze(-1).to(device)
            l = l.unsqueeze(-1).to(device)

            l = (l - mean_loudness) / std_loudness

            y = model(p, l).squeeze(-1)

            ori_stft = multiscale_fft(
                s,
                config["train"]["scales"],
                config["train"]["overlap"],
            )
            rec_stft = multiscale_fft(
                y,
                config["train"]["scales"],
                config["train"]["overlap"],
            )

            loss = 0
            for s_x, s_y in zip(ori_stft, rec_stft):
                lin_loss = (s_x - s_y).abs().mean()
                log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
                loss = loss + lin_loss + log_loss

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            opt.step()

            for g in opt.param_groups:
                g["lr"] = schedule(step)

            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element
            step += 1

            if not step % 100:
                wandb.log({
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": opt.param_groups[0]["lr"],
                    "reverb_decay": torch.nn.functional.softplus(-model.reverb.decay).item() * 500,
                    "reverb_wet": torch.sigmoid(model.reverb.wet).item(),
                    "epoch": e,
                }, step=step)

            if not step % 1000:
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(
                        model.state_dict(),
                        save_path / "state.pth",
                    )
    
                audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()
                
                wandb.log({"mean_loss": mean_loss,
                           "audio": wandb.Audio(audio, sample_rate=config["preprocess"]["sampling_rate"])
                }, step=step)
                
                mean_loss = 0.0
                n_element = 0
                
                sf.write(
                    save_path / f"eval_{e:06d}.wav",
                    audio,
                    config["preprocess"]["sampling_rate"],
                )
        
    run.finish()
