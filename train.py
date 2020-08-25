import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd 
# import torchsummary
import numpy as np 
from dataset import EnergySet, QEnergy, uniEnergySet, hourEnergy
from utils import cv_rmse, cv_rmse_loss, train_loss
from model import CNN, RNN, ensembleNN, concatNN, HourNN, HourRNN, NeuralNet



def train_daily_model(energy, model, mode="time_series"):    
    device = torch.device("cpu")
    # W energy model
    train_dataset = EnergySet("train", mode, energy, True)
    test_dataset = EnergySet("test", mode, energy, True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    # model = NeuralNet(15, 128, 2)
    # model = RNN(19, 256, 3, 1)
    # model = ensembleNN(19, 256, 3, 1)

    # model = CNN(1)
    # torchsummary.summary(model, (24, 9))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    total_step = len(train_dataloader)
    for epoch in range(20):
        for i, (features, labels) in enumerate(train_dataloader):
            # print(i)
            # features = torch.FloatTensor(features)
            outputs = model(features)
            # loss =cv_rmse_loss(outputs, labels, 0.5, 0.5)
            # loss = cv_rmse(outputs, labels)
            loss = cv_rmse(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, 30, i+1, total_step, loss.item())) 

    W_preds = []
    W_targets = []
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = model(features)
            W_preds.append(outputs.numpy())
            W_targets.append(labels.numpy())
        W_preds = np.array(W_preds)
        W_targets = np.array(W_targets)
        W_preds = W_preds.reshape((-1, 1))
        W_targets = W_targets.reshape((-1, 1))
        print(cv_rmse(torch.from_numpy(W_preds), torch.from_numpy(W_targets)))
        print(W_preds[:100])
    torch.save(model.state_dict(), 'dailynn_'  + energy + '.pth')
    
    # Q model
    # train_dataset = EnergySet("train", "time_series", "Q")
    # test_dataset = EnergySet("test", "time_series", "Q")
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    # # model = NeuralNet(15, 128, 2)
    # # model = RNN(19, 256, 3, 1)
    # # model = ensembleNN(19, 256, 3, 1)
    # # model = concatNN(19, 256, 3, 1, "add")
    # model = CNN(1)
    # # torchsummary.summary(model, (24, 9))
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # total_step = len(train_dataloader)
    # for epoch in range(30):
    #     for i, (features, labels) in enumerate(train_dataloader):
    #         # print(i)
    #         # features = torch.FloatTensor(features)
    #         outputs = model(features)
    #         # loss =cv_rmse_loss(outputs, labels, 0.5, 0.5)
    #         # loss = cv_rmse(outputs, labels)
    #         loss = cv_rmse(outputs, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if (i+1) % 10 == 0:
    #             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    #                 .format(epoch+1, 30, i+1, total_step, loss.item())) 

    # Q_preds = []
    # Q_targets = []
    # with torch.no_grad():
    #     for features, labels in test_dataloader:
    #         outputs = model(features)
    #         Q_preds.append(outputs.numpy())
    #         Q_targets.append(labels.numpy())
    #     Q_preds = np.array(Q_preds)
    #     Q_targets = np.array(Q_targets)
    #     Q_preds = Q_preds.reshape((-1, 1))
    #     Q_targets = Q_targets.reshape((-1, 1))
    #     print(cv_rmse(torch.from_numpy(Q_preds), torch.from_numpy(Q_targets)))

    #     # print(cv_rmse_loss(torch.from_numpy(preds).squeeze(1), torch.from_numpy(targets).squeeze(1), 0.3, 0.7))
    #     # print(cv_rmse(targets[:, 0], preds[:, 0]), cv_rmse(targets[:, 1], preds[:, 1]))
    #     # print(test_dataset.label_scaler.inverse_transform(preds))
    #     # print(test_dataset.label_scaler.inverse_transform(test_dataset.test_label))

    #     print(cv_rmse_loss(torch.from_numpy(np.hstack([W_preds, Q_preds])), torch.from_numpy(np.hstack([W_targets, Q_targets]))))
    # torch.save(model.state_dict(), 'dailynn_q.pth')
    
    # print(dataset[0])

def train_hour_model(energy='Q'):
    train_dataset = hourEnergy("train", energy)
    test_dataset = hourEnergy("test", energy)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # model = HourNN()
    model = HourRNN(4, 256, 3, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_step = len(train_dataloader)
    for epoch in range(15):
        for i, ((hour_features, daily_features), labels) in enumerate(train_dataloader):
            # print(i)
            # features = torch.FloatTensor(features)
            outputs = model((hour_features, daily_features))
            # loss =cv_rmse_loss(outputs, labels, 0.5, 0.5)
            # loss = cv_rmse(outputs, labels)
            loss = train_loss(outputs, labels, daily_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, 30, i+1, total_step, loss.item())) 

    Q_preds = []
    Q_targets = []
    with torch.no_grad():
        for (hour_features, daily_features), labels in test_dataloader:
            outputs = model((hour_features, daily_features))
            Q_preds.append(outputs.numpy())
            Q_targets.append(labels.numpy())
        Q_preds = np.array(Q_preds)
        Q_targets = np.array(Q_targets)
        Q_preds = Q_preds.reshape((-1, 1))
        Q_targets = Q_targets.reshape((-1, 1))
        print(cv_rmse(torch.from_numpy(Q_preds), torch.from_numpy(Q_targets)))
        print(Q_preds[:24])
        print(Q_targets[:24])

    torch.save(model.state_dict(), 'hournn_' +energy + '.pth')
if __name__ == "__main__":
    # train_hour_model('W')
    model = NeuralNet(4, 256, 1)
    # model = concatNN(20, 256, 3, 1, "add")
    train_daily_model('Q', model, "regression")
