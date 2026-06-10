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

from ddsp.core import mean_std_loudness
from ddsp.model import DDSP
from ddsp.utils import get_scheduler, spectral_loss
from preprocess import DatasetMultiInstrument


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
    SPLIT_DATASET = True


def make_dataloaders(out_dir, instrument, batch_size, split=True):
    """Return (train, val, test) dataloaders, or (train, None, None) if split=False."""
    train_dataset = DatasetMultiInstrument(out_dir, instrument, subset="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )

    if not split:
        return train_dataloader, None, None

    val_dataset  = DatasetMultiInstrument(out_dir, instrument, subset="val")
    test_dataset = DatasetMultiInstrument(out_dir, instrument, subset="test")

    val_dataloader  = torch.utils.data.DataLoader(val_dataset,  batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


def log_step(model, loss, grad_norm, step, epoch, opt):
    '''Log training metrics to wandb.'''
    wandb.log({
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "lr": opt.param_groups[0]["lr"],
        "reverb_decay": torch.nn.functional.softplus(-model.reverb.decay).item() * 500,
        "reverb_wet": torch.sigmoid(model.reverb.wet).item(),
        "epoch": epoch,
    }, step=step)


def log_checkpoint(model, signal, reconstructed_signal, mean_loss, val_loss, best_loss, step, save_path, config):
    ''''Log evaluation metrics to wandb, save a checkpoint if it's the best so far, and save an audio sample of the reconstruction.'''
    audio = torch.cat([signal, reconstructed_signal], -1).reshape(-1).detach().cpu().numpy()

    log_dict = {
        "mean_loss": mean_loss,
        "audio": wandb.Audio(audio, sample_rate=config["preprocess"]["sampling_rate"]),
    }
    if val_loss is not None:
        log_dict["val_loss"] = val_loss

    wandb.log(log_dict, step=step)
                    
    sf.write(
        save_path / f"eval_{step:06d}.wav",
        audio,
        config["preprocess"]["sampling_rate"],
    )

    loss_to_track = val_loss if val_loss is not None else mean_loss
    if loss_to_track < best_loss:
        best_loss = loss_to_track
        torch.save(model.state_dict(), save_path / "state.pth")
    
    return best_loss


def train_step(model, batch, opt, mean_loudness, std_loudness, config, device):
    '''Perform a training step: compute the loss, backpropagate, and update the model parameters. Return the loss, gradient norm, and reconstructed signal for logging.'''
    signal, pitch, loudness = batch
    signal = signal.to(device)
    pitch = pitch.unsqueeze(-1).to(device)
    loudness = loudness.unsqueeze(-1).to(device)

    loudness = (loudness - mean_loudness) / std_loudness

    reconstructed_signal = model(pitch, loudness).squeeze(-1)

    loss = spectral_loss(signal, reconstructed_signal, config["train"]["scales"], config["train"]["overlap"])
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    opt.step()

    return loss, grad_norm, reconstructed_signal


def evaluate(model, dataloader, mean_loudness, std_loudness, config, device):
    '''Return the average loss on the dataloader, or None if dataloader is None (e.g. no validation set).'''
    if dataloader is None:
        return None

    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            signal, pitch, loudness = batch
            signal = signal.to(device)
            pitch = pitch.unsqueeze(-1).to(device)
            loudness = loudness.unsqueeze(-1).to(device)
            loudness = (loudness - mean_loudness) / std_loudness

            reconstructed_signal = model(pitch, loudness).squeeze(-1)

            total_loss += spectral_loss(signal, reconstructed_signal, config["train"]["scales"], config["train"]["overlap"]).item()
            n += 1

    model.train()
    return total_loss / n if n > 0 else float("inf")



def train(model, dataloaders, opt, schedule, config, save_path, device, total_steps):
    train_dataloader, val_dataloader, _ = dataloaders

    mean_loudness, std_loudness = mean_std_loudness(train_dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    mean_loudness = torch.tensor(mean_loudness, device=device)
    std_loudness = torch.tensor(std_loudness, device=device)

    best_loss = float("inf")
    mean_loss = torch.zeros(1, device=device)
    n_element = 0
    step = 0
    epochs = int(np.ceil(total_steps / len(train_dataloader)))

    for e in tqdm(range(epochs)):
        for batch in train_dataloader:
            loss, grad_norm, reconstructed_signal = train_step(
                model, batch, opt, mean_loudness,
                std_loudness, config, device
            )

            for g in opt.param_groups:
                g["lr"] = schedule(step)

            mean_loss += loss.detach()
            n_element += 1
            step += 1

            if not step % 100:
                log_step(model, loss, grad_norm, step, e, opt)

            if not step % 1000:
                mean_loss_val = (mean_loss / n_element).item()
                mean_loss = torch.zeros(1, device=device)
                n_element = 0

                val_loss = evaluate(model, val_dataloader, mean_loudness, std_loudness, config, device)
 
                signal = batch[0].to(device)
                best_loss = log_checkpoint(
                    model, signal, reconstructed_signal, mean_loss_val, val_loss, best_loss, step, save_path, config
                )
      

def main():
    '''Main training loop'''
    args.parse_args()

    # model config
    with open(args.CONFIG, "r") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    instruments = (
        [args.INSTRUMENT] 
        if args.INSTRUMENT 
        else config["data"]["instruments"]
    )

    for instrument in instruments:
        save_path = pathlib.Path(args.ROOT) / args.NAME / timestamp / instrument
        save_path.mkdir(parents=True, exist_ok=True)

        model = DDSP(**config["model"]).to(device)

        dataloaders = make_dataloaders(
            config["preprocess"]["out_dir"], instrument,
            args.BATCH, split=args.SPLIT_DATASET
        )
        train_dataloader = dataloaders[0]

        run = wandb.init(
            project=args.NAME,
            name=instrument,
            config={**config, "instrument": instrument},
        )

        with open(save_path / "config.yaml", "w") as out_config:
            yaml.safe_dump(config, out_config)

        opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)
        schedule = get_scheduler(
            len(train_dataloader), args.START_LR, args.STOP_LR, args.DECAY_OVER
        )

        train(model, dataloaders, opt, schedule, config, save_path, device, args.STEPS)
 
        run.finish()

if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    main()