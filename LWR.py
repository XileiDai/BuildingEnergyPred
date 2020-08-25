import numpy as np
import lowess
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def get_lwr(record = 'Q', holiday_bool = False):
    file_name = os.listdir('Hour-record/')
    all_data = pd.DataFrame()
    holiday = pd.read_excel('holiday.xlsx')
    for i in file_name:
        if record in i:
            data_i = pd.read_csv('Hour-record/'+i)
            if all_data.shape[0] == 0:
                all_data = data_i
            else:
                all_data.append(data_i)
    holiday['holiday'] = pd.to_datetime(holiday['holiday'], format = '%Y-%m-%d %H:%M:%S')
    all_data['Time'] = pd.to_datetime(all_data['Time'], format = '%Y-%m-%d %H:%M:%S')
    all_data['Hour'] = all_data['Time'].dt.hour
    all_data['day'] = pd.to_datetime(all_data['day'], format = '%Y-%m-%d %H:%M:%S')
    all_data['holiday'] = all_data['day'].isin(holiday['holiday']).values | (all_data['Time'].dt.dayofweek>=5).values
    x = np.array(all_data['Hour'][all_data['holiday'] == holiday_bool])
    x = x[np.argsort(x)]
    y = np.array(all_data['Record'][all_data['holiday'] == holiday_bool])
    y = y[np.argsort(x)]
    record_mean = []
    for i in range(24):
        record_mean.append(np.mean(y[x == i]))

    
    return record_mean


if __name__ == '__main__':
    
    record_mean = get_lwr(record = 'Q', holiday_bool = False)
    plt.plot(range(24), record_mean)
    plt.show()
    a = 1