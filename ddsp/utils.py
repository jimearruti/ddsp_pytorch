import numpy as np

from ddsp.core import multiscale_fft, safe_log


def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (stop_lr / start_lr) ** t
        else:
            return stop_lr
    return schedule


def spectral_loss(original, reconstructed, scales, overlap):
    """Multiscale spectral loss (linear + log) between two waveforms."""
    ori_stft = multiscale_fft(original, scales, overlap)
    rec_stft = multiscale_fft(reconstructed, scales, overlap)

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        loss += (s_x - s_y).abs().mean()
        loss += (safe_log(s_x) - safe_log(s_y)).abs().mean()
    return loss