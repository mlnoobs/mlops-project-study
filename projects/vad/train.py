import torch
from torch.utils.data import DataLoader

import numpy as np

import argparse
import os
import time
from tqdm import tqdm
from collections import defaultdict

from model.models import DNN
from utils.utils import get_config
from utils.logger import Logger
from utils.dataloader import VADDataset

from speechbrain.utils.metric_stats import EER


def prepare_directories_and_logger(output_directory):
    if not os.path.isdir(os.path.join(output_directory, 'summary')):
        os.makedirs(os.path.join(output_directory, 'summary'))
        os.chmod(os.path.join(output_directory, 'summary'), 0o775)
    if not os.path.isdir(os.path.join(output_directory, 'model')):
        os.makedirs(os.path.join(output_directory, 'model'))
        os.chmod(os.path.join(output_directory, 'model'), 0o775)
    logger = Logger(os.path.join(output_directory, 'summary'))
    return logger

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict_model'])
    optimizer.load_state_dict(checkpoint_dict['state_dict_optimizer'])
    return model, optimizer

def save_checkpoint(model, optimizer, filepath):
    torch.save({'state_dict_model': model.state_dict(),
                'state_dict_optimizer': optimizer.state_dict()}, filepath)

def evaluate(epoch, model, logger, config):
    model.eval()
    test_dataset = VADDataset(config['test_data'], config)
    metric_dict = defaultdict(lambda: 0)

    # check EER
    positive_scores = []
    negative_scores = []
    with torch.no_grad():
        for idx in tqdm(range(test_dataset.__len__()//100)):  # EER 너무 오래 걸려서 일부만...
            mels, labels = test_dataset.get_sample_all_positions(idx)
            scores, preds = model.infer(mels)
            for score, label in zip(scores.reshape(-1), labels.reshape(-1)):
                if label == 1:
                    positive_scores.append(float(score))
                if label == 0:
                    negative_scores.append(float(score))
    EER_score, EER_threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    metric_dict['EER'] = EER_score*100
    metric_dict['EER_threshold'] = EER_threshold
    model.threshold = torch.tensor(EER_threshold)

    # check other metrics
    TP = TN = FP = FN = 0
    with torch.no_grad():
        for idx in tqdm(range(test_dataset.__len__())):
            mels, labels = test_dataset.get_sample_all_positions(idx)
            scores, preds = model.infer(mels)
            for pred, label in zip(preds.reshape(-1), labels.reshape(-1)):
                if label == 1:
                    if pred == 1:  # TP는 True positive의 약자로, 실제 True인데, 분류모델에서 예측이 True라고 판단된 경우이다.
                        TP += 1
                    if pred == 0:  # TN는 True negative의 약자로, 실제 False인데, 분류모델에서 예측이 False라고 판단된 경우이다.
                        TN += 1
                if label == 0:
                    if pred == 1:  # FP는 False positive의 약자로, 실제 False인데, 분류모델에서 예측이 True라고 판단된 경우이다.
                        FP += 1
                    if pred == 0:  # FN는 False negative의 약자로, 실제 True인데, 분류모델에서 예측이 False라고 판단된 경우이다.
                        FN += 1
        for idx in tqdm(range(test_dataset.__len__()//100)):  # EER 너무 오래 걸려서 일부만...
            mels, labels = test_dataset.get_sample_all_positions(idx)
            scores, preds = model.infer(mels)
            for score, label in zip(scores.reshape(-1), labels.reshape(-1)):
                if label == 1:
                    positive_scores.append(float(score))
                if label == 0:
                    negative_scores.append(float(score))
    metric_dict['precision'] = TP / (TP + FP)
    metric_dict['recall']    = TP / (TP + FN)
    metric_dict['accuracy']  = (TP + TN) / (TP + TN + FP + FN)
    metric_dict['f1_score']  = 2*(metric_dict['precision']*metric_dict['recall'])/(metric_dict['precision']+metric_dict['recall'])

    # plot 3 samples
    mel_list, label_list, input_mel_list, wavdir_list = test_dataset.get_samples_for_plot(n_samples=3)
    score_list = []
    for mel in input_mel_list:
        score, _ = model.infer(mel)
        score_list.append(score.detach().cpu().numpy())

    logger.log_metric(metric_dict, epoch)    
    logger.log_samples(mel_list, label_list, score_list, wavdir_list, epoch)
    model.train()


def main(output_directory, load_checkpoint_filename, config):
    model = DNN(config['in_features'], config['hidden_features_list'], config['dropout']).to(config['device'])
    print(model)

    logger = prepare_directories_and_logger(output_directory)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'])
    # Load checkpoint if one exists
    if load_checkpoint_filename is not None:
        model, optimizer = load_checkpoint(model, optimizer, load_checkpoint_filename)

    # Training
    loss_fn = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
    train_dataset = VADDataset(config['train_data'], config)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'])
    model.train()
    for epoch in range(config['n_epochs']):
        for batch_idx, (mel, label) in enumerate(train_loader):
            output = model(mel)
            #label = torch.nn.functional.one_hot(label.long()).squeeze(1).float()
            loss = loss_fn(output, label)
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Loss: {:.6f}'.format(epoch, loss.item()))
        if epoch % config['epochs_per_eval'] == 0:
            evaluate(epoch, model, logger, config)
        checkpoint_path = os.path.join(output_directory, 'model', "checkpoint_{}".format(epoch))
        save_checkpoint(model, optimizer, checkpoint_path)
        print("")
    print('train completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='result',
                        help='directory to save checkpoints')
    parser.add_argument('-f', '--load_checkpoint_filename', type=str, default=None,
                        required=False, help='load checkpoint filename')
    parser.add_argument('-c', '--yaml-dir', type=str, default='./config/base.yaml',
                        help="YAML file for config")

    args = parser.parse_args()
    config = get_config(args.yaml_dir, "model")
    print("model:", config['model_name'])

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.manual_seed(config['seed'])

    main(os.path.join(args.output_directory, os.path.splitext(os.path.basename(args.yaml_dir))[0]), args.load_checkpoint_filename, config)

