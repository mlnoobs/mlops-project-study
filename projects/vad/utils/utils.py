import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

def load_wav(data_dir, target_sr):
    wav, sr = librosa.load(data_dir, sr=None)  # sr=None이면 wav header에서 sr을 읽어냄
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)  # 모든 데이터의 sr을 동일하게 맞추기 위함
    return wav

def load_text(filename, split=" "):
    with open(filename, encoding='utf-8') as f:
        text = [line.strip().split(split) for line in f]
    return text

def get_config(filename, key):
    with open(filename) as f:
        docs = yaml.load_all(f, Loader=yaml.Loader)
        config_dict = dict()
        for doc in docs:
            for k, v in doc.items():
                config_dict[k] = v
    return config_dict[key]

def plot_wav_and_label(wav, label_orig, output_filename):
    fig = plt.figure(figsize=(16,9))
    plt.plot(wav)
    plt.plot(label_orig*0.3)
    plt.savefig(os.path.splitext(output_filename)[0] + '.png')
    plt.close()
