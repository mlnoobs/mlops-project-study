import torch
from torch.utils.data import Dataset

import numpy as np
from glob import glob

def process_neighbor(w, u):
    # https://github.com/jtkim-kaist/VAD/blob/master/lib/python/utils.py#L471
    neighbors_1 = np.arange(-w, -u, u)
    neighbors_2 = np.array([-1, 0, 1])
    neighbors_3 = np.arange(1+u, w+1, u)
    neighbors = np.concatenate((neighbors_1, neighbors_2, neighbors_3), axis=0)
    return neighbors

class VADDataset(Dataset):
    def __init__(self, base_dir, config):
        self.metadata_dir_list = glob(base_dir + '/**/*.json', recursive=True)  # 현재 디렉토리를 포함한 모든 하위 디렉토리 탐색
        self.mel_dir_list = [data_dir.replace(".json", "_mel.npy") for data_dir in self.metadata_dir_list]
        self.label_dir_list = [data_dir.replace(".json", "_label.npy") for data_dir in self.metadata_dir_list]
        print("loaded {}".format(base_dir))
        print("{} sentences".format(len(self.mel_dir_list)))

        self.w = config['w']
        self.u = config['u']
        self.neighbors = process_neighbor(self.w, self.u)
        self.device = config['device']

    def __len__(self):
        return len(self.mel_dir_list)

    def get_samples_for_plot(self, n_samples=3):
        mel_list = []
        label_list = []
        input_mel_list = []
        wavdir_list = []
        for idx in np.arange(n_samples):
            mel = np.load(self.mel_dir_list[idx])
            label = np.load(self.label_dir_list[idx])
            input_mel, _ = self.get_sample_all_positions(idx)
            wavdir = self.mel_dir_list[idx].replace('_mel.npy', '.wav')

            mel_list.append(mel)
            label_list.append(label)
            input_mel_list.append(input_mel)
            wavdir_list.append(wavdir)
        return mel_list, label_list, input_mel_list, wavdir_list

    def get_sample_all_positions(self, idx):
        mel = np.load(self.mel_dir_list[idx])
        label = np.load(self.label_dir_list[idx])
        C, L = mel.shape

        # mel과 label의 앞 뒤를 padding
        mel = np.pad(mel, ((0, 0), (self.w, self.w)))
        label = np.pad(label, (self.w, self.w))
        C, L = mel.shape

        all_positions = np.arange(self.w, L-self.w)
        mel_neighbors = [[mel[:, position+n] for n in self.neighbors] for position in all_positions]
        mel_neighbors = np.asarray(mel_neighbors).reshape(-1, C*len(self.neighbors))
        mel = torch.from_numpy(mel_neighbors)

        label = label[all_positions]
        label = torch.from_numpy(label).unsqueeze(-1)
        return mel.to(self.device), label.to(self.device)

    def __getitem__(self, idx):
        # numpy array였던 mel과 label을 불러와서 torch.Tensor로 변환.
        # 임의의 frame index를 중심으로 규칙에 의해 정의되는 주변 neighbors의 mel들을 같이 엮어서 모델이 보도록 함.
        mel = np.load(self.mel_dir_list[idx])
        label = np.load(self.label_dir_list[idx])

        # mel과 label의 앞 뒤를 padding
        mel = np.pad(mel, ((0, 0), (self.w, self.w)))
        label = np.pad(label, (self.w, self.w))
        C, L = mel.shape

        # # 임의의 index를 완전히 랜덤하게 추출?
        # position = np.random.randint(self.w, L-self.w)
        # 임의의 index를 positive와 negative에서 각각 50% 확률로 추출
        # 한 문장 내에 어느 한 쪽 label이 한개도 없는 경우 예외처리
        positive_indices = np.where(label[self.w:L-self.w] == 1)[0] + self.w
        negative_indices = np.where(label[self.w:L-self.w] == 0)[0] + self.w
        if len(negative_indices) == 0:
            position = np.random.choice(positive_indices)
        elif len(positive_indices) == 0:
            position = np.random.choice(negative_indices)
        else:
            if np.random.uniform() > 0.5:
                position = np.random.choice(positive_indices)
            else:
                position = np.random.choice(negative_indices)

        mel_neighbors = [mel[:, position+n] for n in self.neighbors]
        mel_neighbors = np.asarray(mel_neighbors).reshape(-1)
        mel = torch.from_numpy(mel_neighbors)

        label = label[position]
        label = torch.Tensor([label])
        return mel.to(self.device), label.to(self.device)

