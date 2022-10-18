import librosa
import soundfile as sf
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from utils.utils import load_wav, load_text, plot_wav_and_label, get_config
from utils.audio_processing import add_noise, signal_to_melspec


def preprocess_data(noise_wav, signal_wav, signal_phn, snr, config):
    """
    데이터들로 VAD 훈련에서 사용하는 형태로 전처리를 수행하는 함수.

    ARGS
    ------
    noise_wav: 노이즈 데이터. (1-D numpy float array)
    signal_wav: 음성신호 데이터. (1-D numpy float array)
    signal_phn: signal_wav에 대응하는 음소열 정보 데이터. TIMIT 양식만 지원. (string 담고 있는 2-D List)
    snr: signal-to-noise ratio: 음성과 노이즈의 비율을 dB scale로 나타내는 수치. snr 값만큼 signal_wav에 noise_wav를 가한다. (scalar)
    config: YAML 파일에 담긴 Preprocess configurations
    """

    # 깨끗한 원본 음성신호에 노이즈를 임의의 snr값 세기로 가해진 훈련 데이터를 만드는 부분
    # noise의 길이가 signal보다 길다. signal_wav 길이 만큼의 구간을 noise_wav의 임의의 위치에서 추출.
    offset = np.random.randint(0, len(noise_wav)-len(signal_wav)+1)
    noise_clip_wav = noise_wav[offset:offset+len(signal_wav)]
    signal_wav_noise_added = add_noise(signal_wav, noise_clip_wav, snr)

    # noise를 추가한 signal에서 음성 특징벡터: 40-dim, local z-score normalized, log-mel-spectrogram을 뽑는다.
    mel = signal_to_melspec(signal_wav_noise_added,
                            sr=config['sr_model'],
                            n_fft=config['n_fft'],
                            hop_length=config['hop_length'],
                            win_length=config['win_length'],
                            window=config.fn_window(config['win_length']),
                            n_mel_channels=config['n_mel_channels'],
                            mel_fmin=config['mel_fmin'],
                            mel_fmax=config['mel_fmax'])
    # mel을 뽑은 이후에는 혼동되지 않도록 signal_wav_noise_added의 길이를 정확하게 mel 길이의 정수배로 맞춰준다.
    signal_wav_noise_added = signal_wav_noise_added[:mel.shape[-1]*config['hop_length']]
    mel = mel[:signal_wav_noise_added.shape[-1]//config['hop_length']]

    # signal_phn 내에서 symbol_sil_list 안에 포함된 symbol이 발견된다면 silence(0)로 라벨링하고, 나머지는 voice다(1).
    label = np.zeros([mel.shape[-1]], dtype=np.int64)
    label_orig = np.zeros([signal_wav_noise_added.shape[-1]], dtype=np.int64)

    for start_sample, end_sample, symbol in signal_phn:
        if symbol not in config['symbol_sil_list']:
            start_frame = int(int(start_sample)/config['hop_length']+0.5)
            end_frame = int(int(end_sample)/config['hop_length']+0.5)
            label[start_frame:end_frame] = 1
            label_orig[start_frame*config['hop_length']:end_frame*config['hop_length']] = 1

    return signal_wav_noise_added, mel, label, label_orig


def main(noise_dir, signal_dir, output_dir, config):
    # 15개 noise 데이터셋 NOISEX-92 를 순서대로 불러옴.
    noise_wav_dir_list = glob(noise_dir + '/**/*.wav', recursive=True)  # 현재 디렉토리를 포함한 모든 하위 디렉토리 탐색
    for n_idx, noise_wav_dir in enumerate(noise_wav_dir_list):
        noise_name = os.path.splitext(os.path.basename(noise_wav_dir))[0]
        print("[{}/{}] \'{}\' noise data".format(n_idx+1, len(noise_wav_dir_list), noise_name))
        noise_wav = load_wav(noise_wav_dir, target_sr=config['sr_model'])

        # 4620개/1680개(train/test) signal 데이터셋 TIMIT 순서대로 불러오기
        signal_wav_dir_list = glob(signal_dir + '/**/*.WAV', recursive=True)  # TIMIT 데이터셋은 확장자가 대문자로 되어있음
        for signal_wav_dir in tqdm(signal_wav_dir_list):
            signal_wav = load_wav(signal_wav_dir, target_sr=config['sr_model'])
            signal_phn = load_text(signal_wav_dir.replace('.WAV', '.PHN'))  # signal_phn은 [DYNAMIC, 3] shape의 2-D list 자료형.
            # signal_wav 에 noise를 [-snr_db_min, snr_db_max] dB 범위 임의의 정수 SNR로 더한다.
            snr = np.random.randint(config['snr_db_min'], config['snr_db_max']+1)

            # 데이터 전처리 수행
            signal_wav_noise_added, mel, label, label_orig = preprocess_data(noise_wav, signal_wav, signal_phn, snr, config)

            # 전처리 완료된 데이터들을 저장. 출력 디렉토리 구조는 signal_dir 의 구조를 똑같이 따라가면서,
            # TRAIN/TEST 디렉토리 밑에서 noise별로 따로 구분.
            output_filename = os.path.abspath(signal_wav_dir).replace(os.path.abspath(signal_dir), os.path.abspath(output_dir))
            output_filename = output_filename.replace(os.path.abspath(output_dir)+"/TRAIN",
                                                      os.path.abspath(output_dir)+"/TRAIN/"+noise_name)
            output_filename = output_filename.replace(os.path.abspath(output_dir)+"/TEST",
                                                      os.path.abspath(output_dir)+"/TEST/"+noise_name)
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            # 음성 저장
            sf.write(os.path.splitext(output_filename)[0] + '.wav', signal_wav_noise_added, config['sr_model'])
            # 특징벡터 저장
            np.save(os.path.splitext(output_filename)[0] + '_mel.npy', mel)
            # 라벨 저장
            np.save(os.path.splitext(output_filename)[0] + '_label.npy', label)
            # 메타데이터 저장: json
            text = ' '.join(load_text(signal_wav_dir.replace('.WAV', '.TXT'))[0][2:])
            json_object = {
                "noise_name": noise_name,
                "noise_snr_db": snr,
                "text": text,
            }
            with open(os.path.splitext(output_filename)[0] + '.json', 'w') as f:
                json.dump(json_object, f, indent=2)
            ## 라벨링이 잘 되었는지? 확인용 이미지 파일 저장
            #plot_wav_and_label(signal_wav_noise_added, label_orig, output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-yaml', type=str, default='./config/base.yaml',
                        help="YAML file for config")
    parser.add_argument('-n', '--noise-dir', type=str, default='./dataset/noisex92',
                        help="NOISEX-92 dataset directory")
    parser.add_argument('-s', '--signal-dir', type=str, default='./dataset/TIMIT',
                        help="TIMIT dataset directory")
    parser.add_argument('-o', '--output-dir', type=str, default='./preprocessed_dataset/TIMIT',
                        help="preprocessed output directory")
    args = parser.parse_args()

    config = get_config(args.config_yaml, "preprocess")
    np.random.seed(config['seed'])

    main(args.noise_dir, args.signal_dir, args.output_dir, config)

