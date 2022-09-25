import numpy as np


class Config:
    # Audio
    max_wav_value = 32768.0
    sampling_rate = 16000
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    fn_window = np.hanning
    n_mel_channels = 40
    mel_fmin = 0.0
    mel_fmax = sampling_rate/2.0

    chunk_size = 20  # n_frame


class PreprocessConfig(Config):
    sr_model = Config.sampling_rate
    sr_timit = 16000
    sr_noisex = 19980

    symbol_sil_list = ['bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'pau', 'epi', 'h#']

    snr_db_min = -10
    snr_db_max = 12


class DNNConfig(Config):
    batch_size = 32
    n_epochs = 10


def get_config(mode="DNN"):
    return {
        "preprocess": PreprocessConfig,
        "dnn": DNNConfig,
    }[mode]

