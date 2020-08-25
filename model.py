import torch
import torch.nn as nn
import torchsummary

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)[:, -1, :]
        return out

class HourRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HourRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x[0].size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x[0].size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x[0], (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        for i in range(out.size(0)):
            out[i] = out[i] * x[1][i]
        return out    

class HourNN(nn.Module):
    def __init__(self):
        super(HourNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*6*1, 24),
            nn.ReLU()
        )
        # self.fc = nn.Linear(32*6*1+5, 24)
    
    def forward(self, x):
        # 1*24*5
        weather = x[0].unsqueeze(1)
        w_feature = self.layer1(weather)
        w_feature = self.layer2(w_feature)
        w_feature = w_feature.reshape(w_feature.size(0), -1)
        # feature = torch.cat([w_feature, x[1].squeeze(1)], dim=1)
        out = self.fc(w_feature)
        out = out.unsqueeze(-1)
        for i in range(out.size(0)):
            out[i] = out[i] * x[1][i]
        return out
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*5*18, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class ensembleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ensembleNN, self).__init__()    
        self.RNN = RNN(input_size, hidden_size, num_layers, num_classes)
        self.CNN = CNN(num_classes)
    
    def forward(self, x):
        out1 = self.RNN(x)
        out2 = self.CNN(x)
        out = (out1 + out2) / 2
        return out

class concatNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, type='cat'):
        super(concatNN, self).__init__()    
        self.type = type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn_fc = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(32*5*18, hidden_size),
            # nn.ReLU()
        )        
        if self.type == "cat":
            self.fc = nn.Linear(hidden_size*2, num_classes)
        elif self.type == "add":
            self.fc = nn.Linear(hidden_size, num_classes)

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        rnn_feature = self.rnn_fc(out)[:, -1, :]

        x = x.unsqueeze(1)
        out = self.cnn_layer1(x)
        out = self.cnn_layer2(out)
        out = out.reshape(out.size(0), -1)
        cnn_feature = self.cnn_fc(out)
        if self.type == "cat":
            feature = torch.cat([rnn_feature, cnn_feature], dim=1)
        elif self.type == "add":
            feature = rnn_feature + cnn_feature
        out = self.fc(feature)
        out = self.relu(out)
        return out  

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_preds):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 =  nn.Linear(hidden_size//2, num_preds) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
# model = DailyNN(1)
# model = CNN(1)
# torchsummary.summary(model, (7, 20))