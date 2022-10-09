import torch

class DNN(torch.nn.Module):
    def __init__(self, in_features, hidden_features_list, dropout=0.0, threshold_init=0.5):
        super(DNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        prev_features = in_features
        for hidden_features in hidden_features_list:
            self.layers.append(
                torch.nn.Sequential(
                   torch.nn.Linear(prev_features, hidden_features),
                   torch.nn.BatchNorm1d(hidden_features),
                   torch.nn.ReLU(),
                   torch.nn.Dropout(dropout)
                )
            )
            prev_features = hidden_features
        self.layers.append(torch.nn.Linear(prev_features, 1))
        self.register_buffer('threshold', torch.tensor(threshold_init))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def infer(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x, torch.where(x > self.threshold, 1, 0)
        #x = torch.nn.functional.softmax(x, dim=-1)
        #return x[...,1:]
