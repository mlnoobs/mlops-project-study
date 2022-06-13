import os
import syss
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import time

class ModelV3(torch.nn.Module):
    def __init__(self):
        super(ModelV3, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


HOME_DIR = os.path.join(os.getcwd(), os.pardir, os.pardir)
MODEL_DIR = os.path.join(HOME_DIR, 'model.pt')
MNIST_DATA_DIR = os.path.join(HOME_DIR, 'mnist_inputs.npy')

# ref : https://github.com/floydhub/mnist/blob/953589b3466decbaa85fdde3aebdb84489c353e6/ConvNet.py#L72
class MnistInference():
    def __init__(self):
        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._cuda else "cpu")
        self._model = torch.load(MODEL_DIR)

    # inference가 mnist 이미지 파일을 하나 받았을 때 모델에 넣은 결과 (=이미지에 해당하는 숫자) 를 리턴
    # INPUT numpy: mnist image
    # OUTPUT int : prediction using model  
    def inference(self, input_image):
        # TODO: If you use GPU, you should get data from cpu to gpu device
        if self._cuda:
            #self._model.load_state_dict(torch.load(self._ckp))
            pass
        else:
            # Load GPU model on CPU
            #self._model.load_state_dict(torch.load(self._ckp, map_location=lambda storage, loc: storage))
            pass
        preprocessed_input = image_preprocessing(input_image)
        return None
    
    def image_preprocessing(self):
        return None


def get_model():
    return torch.load(MODEL_DIR)

def get_mnist_data():
    return numpy.load(MNIST_DATA_DIR)



if __name__ == "__main__":
    input_data = sys.argv[1]

    mnist_inference = MnistInference()
    pred = MnistInference.inference(input_data)
    print(pred)
