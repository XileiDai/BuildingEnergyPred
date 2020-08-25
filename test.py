import torch
import pandas as pd 
import numpy as np
from dataset import EnergySet, hourEnergy, testingBuildingSet
from torch.utils.data import Dataset, DataLoader
from model import HourNN, HourRNN,CNN, concatNN, RNN,NeuralNet
from utils import cv_rmse


def test(daily_model, hour_model, energy, mode, building_id, year, to_csv=False):
    # if energy == "Q" and separate_feature:
        # test_dataset = EnergySet("test", "regression", energy, separate_feature=separate_feature)
        # test_dataset = testingBuildingSet()
    # else:
        # test_dataset = EnergySet("test", "time_series", energy, separate_feature=separate_feature)
    test_dataset = testingBuildingSet(energy,mode=mode, building_id=building_id, year=year)


    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    daily_preds = []
    daily_targets = []
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = daily_model(features)
            daily_preds.append(outputs.numpy())
            daily_targets.append(labels.numpy())
        daily_preds = np.array(daily_preds)
        daily_targets = np.array(daily_targets)
        daily_preds = daily_preds.reshape((-1, 1))
        daily_targets = daily_targets.reshape((-1, 1))
        print(daily_preds[:30])
        print(daily_targets[:30])
        # add code to output prediction to csv file
        print(cv_rmse(torch.from_numpy(daily_preds), torch.from_numpy(daily_targets)))    
        
        # print(len(daily_preds))
        if to_csv:
            df = pd.read_csv("output_" + energy + ".csv")
            df['prediction'] = daily_preds
            df.to_csv("output_" + energy + ".csv", index=False)



    # hour_dataset = hourEnergy("test", energy)
    # test_dataloader = DataLoader(hour_dataset, batch_size=1, shuffle=False, pin_memory=True)
    # Q_preds = []
    # Q_targets = []
    # with torch.no_grad():
    #     for i, ((hour_features, daily_features), labels) in enumerate(test_dataloader):
    #         if i >= 7:
    #             if separate_feature:
    #                 outputs = hour_model((hour_features, daily_preds[i]))
    #             else:
    #                 outputs = hour_model((hour_features, daily_preds[i-7]))
    #             Q_preds.append(outputs.numpy())
    #             Q_targets.append(labels.numpy())
    #     Q_preds = np.array(Q_preds)
    #     Q_targets = np.array(Q_targets)
    #     Q_preds = Q_preds.reshape((-1, 1))
    #     Q_targets = Q_targets.reshape((-1, 1))
    #     print(cv_rmse(torch.from_numpy(Q_preds), torch.from_numpy(Q_targets)))    
    #     print(Q_preds[:24])
    #     print(Q_targets[:24])

if __name__ == "__main__":
    # daily_model = concatNN(20, 256, 3, 1, "add")
    # daily_model.load_state_dict(torch.load('dailynn_W.pth'))
    # hour_model = HourRNN(5, 256, 3, 1)
    # # hour_model = HourNN()
    # hour_model.load_state_dict(torch.load('hournn_W.pth'))
    # test(daily_model, hour_model, "W")



    daily_model = NeuralNet(5, 256, 1)
    daily_model.load_state_dict(torch.load('dailynn_Q.pth'))

    hour_model = HourRNN(5, 256, 3, 1)
    hour_model.load_state_dict(torch.load('hournn.pth'))
    test(daily_model, hour_model, "Q","regression",  2, 2017)

    # daily_model = CNN(1)
    # daily_model.load_state_dict(torch.load('dailynn_q_CNN.pth'))
    # hour_model = HourRNN(5, 256, 3, 1)
    # hour_model.load_state_dict(torch.load('hournn.pth'))
    # test(daily_model, hour_model, "Q","time_series",  2, 2017)
    