from torch.utils.tensorboard import SummaryWriter
import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_sample(mel, label, score, wavdir):
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.imshow(mel, origin='lower')
    plt.plot(label*30+5, 'r', label='label')
    plt.plot(score*30+5, 'b', label='score', linestyle='--')
    plt.legend()
    ax.set_title(wavdir)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    return fig

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_metric(self, metric_dict, epoch):
        for key, value in metric_dict.items():
            self.add_scalar('metric/'+key, value, epoch)

    def log_samples(self, mel_list, label_list, score_list, wavdir_list, epoch):
        for idx, (mel, label, score, wavdir) in enumerate(zip(mel_list, label_list, score_list, wavdir_list)):
            self.add_figure('samples/plot_{}'.format(idx), plot_sample(mel, label, score, wavdir), epoch)
            wav, sr = librosa.load(wavdir, sr=None)
            self.add_audio('samples/audio_{}'.format(idx), wav, epoch, sample_rate=sr)

