import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchsummary
from utils import cv_rmse_loss, cv_rmse
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from dataset import EnergySet



class Encoder(nn.Module):        
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, (hidden, cell) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, hid_dim, n_layers):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = 0.5, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        # input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        # embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [-1, 24, 9]
        #trg = [-1, 24, 1]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0] #[-1, 24, ]
        trg_len = trg.shape[1]
        trg_output_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_output_size).to(self.device) #[-1, 24, 2]
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = torch.zeros(batch_size, 1, trg_output_size)
        
        for t in range(trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:,t:t+1, :] = output
            
            #decide if we are going to use teacher forcing or not
            # teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            # top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = output
        
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


device = torch.device("cpu")
INPUT_DIM = 15
OUTPUT_DIM = 2
ENC_EMB_DIM = 128
DEC_EMB_DIM = 2
HID_DIM = 128
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS)
# torchsummary.summary(enc, (24, 9))
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)
print(model)

train_dataset = EnergySet("train", "time_series")
# print(train_dataset[0])
test_dataset = EnergySet("test", "time_series")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
# print(len(train_dataset), len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
total_step = len(train_dataloader)
for epoch in range(150):
    for i, (features, labels) in enumerate(train_dataloader):
        # features = torch.FloatTensor(features)
        outputs = model(features, labels)
        loss =cv_rmse_loss(outputs, labels, 0.2, 0.8)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item())) 

with torch.no_grad():
    preds = []
    targets = []
    for features, labels in test_dataloader:
        outputs = model(features, labels)
        preds.append(outputs.numpy())
        targets.append(labels.numpy())
        # loss = cv_rmse_loss(outputs, labels)
        # print(loss.item())
    
    preds = np.array(preds)
    preds = preds.reshape((-1, 2))
    targets = np.array(targets)
    targets = targets.reshape((-1, 2))
    print(cv_rmse(targets[:, 0], preds[:, 0]), cv_rmse(targets[:, 1], preds[:, 1]))

    print(cv_rmse_loss(torch.from_numpy(preds), torch.from_numpy(targets), 0.3, 0.7))
    
    # mat = np.zeros((len(preds), 10))
    # mat[:, 8] = preds[:, 0]
    # mat[:, 9] = preds[:, 1]
    print(preds)