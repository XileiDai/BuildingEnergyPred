import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd 
import numpy as np

class EnergySet(Dataset):
    def __init__(self, phase = "train",  type="regression", energy="Q", separate_feature=False):
        # self.df = pd.read_csv("all_buildings.csv")
        # self.transform = transform
        self.phase = phase
        self.type = type
        self.energy = energy
        self.separate_feature = separate_feature
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_csv("all_buildings_daily.csv")
        df['BuildingID'] = df['BuildingID'].astype('int64')
        # df = df.drop(["Time", "BuildingID"], axis=1)
        if self.separate_feature:
            if self.energy == "W":
                scale_feature = ['tem_avg','tem_max', 'Stair1', 'Stair2', 'Area', 'HVACType', \
                    'tem_min','tem_dew_avg','tem_dew_max','tem_dew_min','rh_avg','rh_max','rh_min','pressure_avg','pressure_max','pressure_min','vel_avg','vel_max','vel_min']
            elif self.energy == "Q":
                scale_feature = ['Stair1', 'Stair2', 'Area']
        else:
            scale_feature = ['tem_avg','tem_max', 'Stair1', 'Stair2', 'Area', 'HVACType', \
                    'tem_min','tem_dew_avg','tem_dew_max','tem_dew_min','rh_avg','rh_max','rh_min','pressure_avg','pressure_max','pressure_min','vel_avg','vel_max','vel_min']            
        self.scaler = StandardScaler()
        self.scaler.fit(df[scale_feature])
        df[scale_feature] = self.scaler.transform(df[scale_feature])
        self.train = df[df["BuildingID"].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15'])]
        self.test = df[df["BuildingID"].isin(['18'])]
        self.train_feature = self.train[scale_feature+['holiday']].values 
        self.test_feature = self.test[scale_feature+['holiday']].values 

        if self.energy == "Q":
            self.train_label = self.train[['Q_avg']].values
            self.test_label = self.test[['Q_avg']].values    
        elif self.energy == "W":
            self.train_label = self.train[['W_avg']].values
            self.test_label = self.test[['W_avg']].values
        elif self.energy == "A":
            self.train_label = self.train[['W_avg', 'Q_avg']].values
            self.test_label = self.test[['W_avg', 'Q_avg']].values                    
        # self.label_scaler = MinMaxScaler()
        # self.label_scaler.fit(self.train_label)
        # self.train_label = self.label_scaler.transform(self.train_label)
        # self.test_label = self.label_scaler.transform(self.test_label)    
    
    def __len__(self):
        if self.phase == "train":
            if self.type == "regression":
                return len(self.train_feature)
            elif self.type == "time_series":
                return len(self.train_feature) - 7
        else:
            if self.type == "regression":
                return len(self.test_feature)
            elif self.type == "time_series":
                return len(self.test_feature) - 7 
    
    def __getitem__(self, idx):
        if self.phase == "train":
            if self.type == "regression":
                return torch.from_numpy(self.train_feature[idx]).float(), torch.from_numpy(self.train_label[idx]).float()
            elif self.type == "time_series":
                return torch.from_numpy(self.train_feature[idx:(idx+7)]).float(), \
                                        torch.from_numpy(self.train_label[(idx+7)]).float()
        else:
            if self.type == "regression":
                return torch.from_numpy(self.test_feature[idx]).float(), torch.from_numpy(self.test_label[idx]).float()
            elif self.type == "time_series":
                return torch.from_numpy(self.test_feature[idx:(idx+7)]).float(), \
                                        torch.from_numpy(self.test_label[(idx+7)]).float()

class testingBuildingSet(Dataset):
    def __init__(self, energy, mode="time_series", building_id=20,year=2017):
        self.energy = energy
        self.mode = mode
        self.reference_df = pd.read_csv("all_buildings_daily.csv")
        # self.hour_df = pd.read_csv("all_buildings.csv")
        self.scale_feature = ['tem_avg','tem_max', 'Stair1', 'Stair2', 'Area', 'HVACType', \
            'tem_min','tem_dew_avg','tem_dew_max','tem_dew_min','rh_avg','rh_max','rh_min','pressure_avg','pressure_max','pressure_min','vel_avg','vel_max','vel_min']        
        self.scaler = StandardScaler()
        self.scaler.fit(self.reference_df[self.scale_feature])
        if building_id == 20:
            test_df = pd.read_csv("testing_building.csv")
            test_df[self.scale_feature] = self.scaler.transform(test_df[self.scale_feature])    
            self.label = np.zeros(365)
        else:
            self.reference_df['BuildingID'] = self.reference_df['BuildingID'].astype('int64')
            self.reference_df['Time'] = pd.to_datetime(self.reference_df['Time'])
            test_df = self.reference_df[self.reference_df['BuildingID'] == building_id]
            test_df = test_df[test_df['Time'] > pd.datetime(year - 1, 12, 24)]
            test_df[self.scale_feature] = self.scaler.transform(test_df[self.scale_feature])    
            self.label = test_df[energy + '_avg'].values[7:]

        if self.mode == "time_series":
            self.q_feature = test_df[self.scale_feature + ['holiday']].values
        else:
            self.q_feature = test_df[['Stair1', 'Stair2', 'Area', 'HVACType', 'holiday']].values 
        
        self.w_feature = test_df[self.scale_feature + ['holiday']].values

    def __len__(self):
        return 365

    def __getitem__(self, idx):
        if self.mode == "time_series" and self.energy == "Q":
            return torch.from_numpy(self.q_feature[idx:idx+7]).float(), torch.tensor(self.label[idx]).float()
        elif self.mode == "regression" and self.energy == "Q":
            return torch.from_numpy(self.q_feature[idx+7]).float(), torch.tensor(self.label[idx]).float()
        elif self.mode == "time_series" and self.energy == "W":
            return torch.from_numpy(self.w_feature[idx:idx+7]).float(), torch.tensor(self.label[idx]).float()


class hourEnergy(Dataset):
    def __init__(self, phase = "train", energy="Q"):
        # self.df = pd.read_csv("all_buildings.csv")
        # self.transform = transform
        self.phase = phase
        self.energy = energy
        self.data_preprocessing()

    def data_preprocessing(self):
        df_daily = pd.read_csv("all_buildings_daily.csv")
        df_daily['BuildingID'] = df_daily['BuildingID'].astype('int64')
        df_hour = pd.read_csv("all_buildings.csv")
        scale_feature = ['temp','point_temp','humidity','pa','wind speed']
        # df = df.drop(["Time", "BuildingID"], axis=1)
        # scale_feature = ['tem_avg','tem_max', 'Stair1', 'Stair2', 'Area', 'HVACType', \
        # 'tem_min','tem_dew_avg','tem_dew_max','tem_dew_min','rh_avg','rh_max','rh_min','pressure_avg','pressure_max','pressure_min','vel_avg','vel_max','vel_min']
        self.scaler = StandardScaler()
        self.scaler.fit(df_hour[scale_feature])
        df_hour[scale_feature] = self.scaler.transform(df_hour[scale_feature])
        scale_feature = ['Stair1','Stair2','Area']
        self.scaler2 = StandardScaler()
        self.scaler2.fit(df_daily[scale_feature])
        df_daily[scale_feature] = self.scaler2.transform(df_daily[scale_feature])
        self.train_hour = df_hour[df_hour["BuildingID"].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15'])]
        self.test_hour = df_hour[df_hour["BuildingID"].isin(['18'])]
        self.train_daily = df_daily[df_daily["BuildingID"].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '15'])]
        self.test_daily = df_daily[df_daily["BuildingID"].isin(['18'])]
        # self.train_feature = self.train[scale_feature].values 
        # self.test_feature = self.test[scale_feature].values 

        # if self.energy == "Q":
        #     self.train_label = self.train[['Q_avg']].values
        #     self.test_label = self.test[['Q_avg']].values    
        # elif self.energy == "W":
        #     self.train_label = self.train[['W_avg']].values
        #     self.test_label = self.test[['W_avg']].values
        # elif self.energy == "A":
        #     self.train_label = self.train[['W_avg', 'Q_avg']].values
        #     self.test_label = self.test[['W_avg', 'Q_avg']].values

    def __len__(self):
        if self.phase == "train":
            return len(self.train_daily)
        else:
            return len(self.test_daily)

    def  __getitem__(self, idx):
        if self.phase == "train":
            hour_weather = self.train_hour.iloc[idx:idx+24][['temp','point_temp','humidity','pa','wind speed']].values
            if self.energy == "Q":
                daily_feature = self.train_daily.iloc[idx][['Q_avg']].values.reshape(1, -1).astype(np.float64)
                hour_label = self.train_hour.iloc[idx:idx+24][['Record_Q']].values
            elif self.energy == "W":
                hour_label = self.train_hour.iloc[idx:idx+24][['Record_W']].values
                daily_feature = self.train_daily.iloc[idx][[ 'W_avg']].values.reshape(1, -1).astype(np.float64)
            elif self.energy == "A":
                hour_label = self.train_hour.iloc[idx:idx+24][['Record_W','Record_Q']].values
                daily_feature = self.train_daily.iloc[idx][[ 'Q_avg', 'W_avg']].values.reshape(1, -1).astype(np.float64)
        else:
            hour_weather = self.test_hour.iloc[idx:idx+24][['temp','point_temp','humidity','pa','wind speed']].values
            if self.energy == "Q":
                daily_feature = self.test_daily.iloc[idx][['Q_avg']].values.reshape(1, -1).astype(np.float64)
                hour_label = self.test_hour.iloc[idx:idx+24][['Record_Q']].values
            elif self.energy == "W":
                hour_label = self.test_hour.iloc[idx:idx+24][['Record_W']].values
                daily_feature = self.test_daily.iloc[idx][[ 'W_avg']].values.reshape(1, -1).astype(np.float64)
            elif self.energy == "A":
                hour_label = self.test_hour.iloc[idx:idx+24][['Record_W','Record_Q']].values
                daily_feature = self.test_daily.iloc[idx][[ 'Q_avg', 'W_avg']].values.reshape(1, -1).astype(np.float64)            
        return (torch.from_numpy(hour_weather).float(), torch.from_numpy(daily_feature).float()), torch.from_numpy(hour_label).float()

class QEnergy(Dataset):
    def __init__(self, phase = "train",  type="regression"):
        # self.df = pd.read_csv("all_buildings.csv")
        # self.transform = transform
        self.phase = phase
        self.type = type
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_csv("all_buildings_daily_Q.csv")
        df['BuildingID'] = df['BuildingID'].astype('int64')
        # df = df.drop(["Time", "BuildingID"], axis=1)
        scale_feature = ['tem_avg','tem_max', 'Stair1', 'Stair2', 'Area', 'HVACType', \
        'tem_min','tem_dew_avg','tem_dew_max','tem_dew_min','rh_avg','rh_max','rh_min','pressure_avg','pressure_max','pressure_min','vel_avg','vel_max','vel_min']
        self.scaler = StandardScaler()
        self.scaler.fit(df[scale_feature])
        df[scale_feature] = self.scaler.transform(df[scale_feature])
        self.train = df[df["BuildingID"].isin([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 7, 15])]
        self.test = df[df["BuildingID"] == 18]
        self.train_feature = self.train[scale_feature].values 
        self.train_label = self.train[['Q_avg']].values
        # self.label_scaler = MinMaxScaler()
        # self.label_scaler.fit(self.train_label)
        # self.train_label = self.label_scaler.transform(self.train_label)
        self.test_feature = self.test[scale_feature].values 
        self.test_label = self.test[['Q_avg']].values    
        # self.test_label = self.label_scaler.transform(self.test_label)    
    
    def __len__(self):
        if self.phase == "train":
            if self.type == "regression":
                return len(self.train_feature)
            elif self.type == "time_series":
                return len(self.train_feature) - 15
        else:
            if self.type == "regression":
                return len(self.test_feature)
            elif self.type == "time_series":
                return len(self.test_feature) - 15 
    
    def __getitem__(self, idx):
        if self.phase == "train":
            if self.type == "regression":
                return torch.from_numpy(self.train_feature[idx]).float(), torch.from_numpy(self.train_label[idx]).float()
            elif self.type == "time_series":
                return torch.from_numpy(self.train_feature[idx:(idx+15)]).float(), \
                                        torch.from_numpy(self.train_label[idx:(idx+15)]).float()
        else:
            if self.type == "regression":
                return torch.from_numpy(self.test_feature[idx]).float(), torch.from_numpy(self.test_label[idx]).float()
            elif self.type == "time_series":
                return torch.from_numpy(self.test_feature[idx:(idx+15)]).float(), \
                                        torch.from_numpy(self.test_label[idx:(idx+15)]).float()

class uniEnergySet(Dataset):
    def __init__(self, phase = "train", energy="Q", seq_len=7):
        # self.df = pd.read_csv("all_buildings.csv")
        # self.transform = transform
        self.phase = phase
        self.energy = energy
        self.seq_len = seq_len
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_csv("all_buildings_daily.csv")
        df['BuildingID'] = df['BuildingID'].astype('int64')
        # df = df.drop(["Time", "BuildingID"], axis=1)
        self.train = df[df["BuildingID"].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '7', '15'])]
        self.test = df[df["BuildingID"].isin(['17', '18'])]
        if self.energy == "Q":
            self.train = self.train[['Q_avg']].values
            self.test = self.test[['Q_avg']].values    
        else:
            self.train = self.train[['W_avg']].values
            self.test = self.test[['W_avg']].values              
    
    def __len__(self):
        if self.phase == "train":
            return len(self.train) - self.seq_len
        else:
            return len(self.test) - self.seq_len
    
    def __getitem__(self, idx):
        if self.phase == "train":
            return torch.from_numpy(self.train[idx:(idx+self.seq_len)]).float(), \
                                        torch.from_numpy(self.train[idx+self.seq_len:idx+self.seq_len+1]).float()
        else:
            return torch.from_numpy(self.test[idx:(idx+self.seq_len)]).float(), \
                                        torch.from_numpy(self.test[idx+self.seq_len:idx+self.seq_len+1]).float()
# parameter setting
# 3days, RNN(15, 128, 3, 2), loss: (0.2, 0.8), lr: 0.005, epochL 200
                            