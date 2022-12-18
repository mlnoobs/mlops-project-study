import torch
import argparse
import numpy as np

from model.models import DNN
from utils.audio_processing import signal_to_melspec
from utils.utils import load_wav, get_config
from utils.dataloader import process_neighbor


def load_checkpoint(model, checkpoint_path):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict_model'])
    return model

def get_sample_all_positions(config, mel_file):
    neighbors = process_neighbor(config["p"], config["f"])

    if isinstance(mel_file, str):
        mel = np.load(mel_file)
    else:
        mel = mel_file
    
    mel = np.pad(mel, ((0, 0), (config["p"], config["f"])))
    C, L = mel.shape

    all_positions = np.arange(config["p"], L-config["f"])
    mel_neighbors = [[mel[:, position+n] for n in neighbors] for position in all_positions]
    mel_neighbors = np.asarray(mel_neighbors).reshape(-1, C*len(neighbors))
    mel = torch.from_numpy(mel_neighbors)

    return mel.to(config["device"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--yaml-dir', type=str, default='./config/base.yaml',
                        help="YAML file for config")
    parser.add_argument('-cp', '--check-point', type=str, default='./files/checkpoint_19',
                        help="Checkpoint file for model")
    parser.add_argument('-i', '--input-audio', type=str, default='./files/SI499.wav',
                        help="Audio file for input")
    args = parser.parse_args()
    
    model_config = get_config(args.yaml_dir, "model")
    preprocess_config = get_config(args.yaml_dir, "preprocess")
    load_checkpoint_file = args.check_point
    input_audio_file = args.input_audio

    model = DNN(model_config['in_features'], model_config['hidden_features_list'], model_config['dropout']).to(model_config['device'])

    if load_checkpoint_file is not None:
        model = load_checkpoint(model, load_checkpoint_file)
    

    signal_wav = load_wav(input_audio_file, target_sr=preprocess_config['sr_model'])
    mel = signal_to_melspec(signal_wav,
                            sr=preprocess_config['sr_model'],
                            n_fft=preprocess_config['n_fft'],
                            hop_length=preprocess_config['hop_length'],
                            win_length=preprocess_config['win_length'],
                            window=preprocess_config['fn_window'],
                            n_mel_channels=preprocess_config['n_mel_channels'],
                            mel_fmin=preprocess_config['mel_fmin'],
                            mel_fmax=preprocess_config['mel_fmax'])

    signal_wav = signal_wav[:mel.shape[-1]*preprocess_config['hop_length']]
    mel = mel[:signal_wav.shape[-1]//preprocess_config['hop_length']]

    mels = get_sample_all_positions(model_config, mel)
    scores, preds = model.infer(mels)
    TP = TN = FP = FN = 0
    for pred in zip(preds.reshape(-1)):
        print(pred)