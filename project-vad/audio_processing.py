import librosa
import numpy as np


def linear_to_db(signal):
    return 10*np.log10(signal)

def db_to_linear(signal):
    return 10**(signal/10)

def snr_linear(signal, noise):
    return sum(signal**2) / sum(noise**2)  # signal-to-noise ratio

def snr_db(signal, noise):
    return linear_to_db(snr_linear(signal, noise))  # SNR in dB scale

def add_noise(signal, noise, target_snr):
    # ensure zero mean noise
    noise = noise - np.average(noise)

    # ensure unity standard deviation noise
    noise = noise / np.std(noise)

    # determine the power of the speech signal
    target_snr_linear = db_to_linear(target_snr)
    curr_snr_linear = snr_linear(signal, noise)

    rescaled_noise = noise * np.sqrt(curr_snr_linear/target_snr_linear)

    return signal + rescaled_noise

def z_score_normalize(signal):
    return (signal - np.average(signal)) / np.std(signal)

def signal_to_melspec(signal, sr, n_fft, hop_length, win_length, window, n_mel_channels, mel_fmin, mel_fmax):
    stft = librosa.stft(signal,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window)
    stft = np.clip(np.abs(stft), a_min=1e-10, a_max=None)
    mel_basis = librosa.filters.mel(sr=sr,
                                    n_fft=n_fft,
                                    n_mels=n_mel_channels,
                                    fmin=mel_fmin,
                                    fmax=mel_fmax)

    mel = mel_basis @ stft
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = z_score_normalize(mel)
    return mel

