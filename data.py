import pandas as pd
import datetime
import numpy as np

def add_holiday():
    df_all = pd.read_csv("all_buildings_daily.csv")
    print(len(df_all))
    df_holiday = pd.read_csv("Daily-record/ID-4-Q.csv")
    holiday = df_holiday.drop([424])[['holiday']].values
    holiday = np.tile(holiday, (19, 1))
    print(holiday.shape)
    # df_holiday = df_holiday.reset_index()
    df_all["holiday"] = holiday
    df_all.to_csv('all_buildings_daily.csv', index=False)

def create_testing_building_data():
    df = create_daily_weather()
    reference_df = pd.read_csv('all_buildings_daily.csv')
    df['BuildingID'] = 20
    df['Holiday'] = reference_df['holiday']
    df['Q_avg'] = 0
    df['Q_max'] = 0
    df['Q_min'] = 0
    df['W_avg'] = 0
    df['W_max'] = 0
    df['W_min'] = 0   
    df['Stair1'] = 30
    df['Stair2'] = 4
    df['Area'] = 101816
    df['HVACType'] = 0
    df['Time'] = pd.to_datetime(df['Time'])
    df = df[df['Time'] > pd.datetime(2016, 12, 24)] 
    df.to_csv("testing_building.csv", index=False)
    df[df['Time'] > pd.datetime(2016, 12, 31)]['Time'].to_csv("output_W.csv", index=False)
    df[df['Time'] > pd.datetime(2016, 12, 31)]['Time'].to_csv("output_Q.csv", index=False) 

def create_daily_weather():
    weather = []
    for i in range(2015, 2018):
        df = pd.read_excel(open('weather/shanghai weather_' + str(i) + '.xlsx', 'rb'), sheet_name="Sheet2",  \
                                names=["Time", "temp", "point_temp", "humidity", "pa", "wind speed"])
        df.Time = pd.to_datetime(df.Time).dt.strftime('%Y-%m-%d')
        df_daily = df.groupby(['Time']).apply(f_weather)
        df_daily = df_daily.reset_index()
        df_daily.Time = pd.to_datetime(df_daily.Time)

        weather.append(df_daily)
        # df_daily_mean.rename(columns={'temp': 'temp_avg', 'point_temp': 'tem_dew_avg', 'humidity': 'rh_avg'})
        # df_daily_max
        # print(df_daily)
    weather_df = pd.concat(weather)
    print(weather_df)
    return weather_df

def create_daily_energy_Q():
    buildings = []
    weather_df = create_daily_weather()
    building_info_df = pd.read_excel(open("5 train_building_info.xlsx", "rb"), sheet_name="Sheet1" )
    for i in [0, 1, 3, 4, 5, 7, 8, 9, 10, 14, 15, 18]:
        building = pd.read_csv("./Daily-record/ID-" + str(i) + '-Q.csv', \
            names=["i", "Time", "Q_avg", "Q_min", "Q_max", "holiday", "count"], header=None)
        building = building.dropna()
        # building = building.drop([0])
        building = building.reset_index()
        building['BuildingID'] = i
        building['Stair1'] = building_info_df.iloc[i]['Stair1']
        building['Stair2'] = building_info_df.iloc[i]['Stair2']
        building['Area'] = building_info_df.iloc[i]['Area']
        building['HVACType'] = building_info_df.iloc[i]['HVACType_num']    
        building.Time = pd.to_datetime(building.Time)
        building_1 = building[building['Time'] < pd.datetime(2016,2,29)]
        building_2 = building[building['Time'] >= pd.datetime(2016,3,1)]
        building = pd.concat([building_1, building_2])
        buildings.append(pd.merge(building, weather_df, on='Time'))
        # buildings.append(building)
    all_buildings = pd.concat(buildings)
    all_buildings.to_csv("all_buildings_daily_Q.csv", index=False)

def create_daily_energy_W():
    buildings = []
    weather_df = create_daily_weather()
    building_info_df = pd.read_excel(open("5 train_building_info.xlsx", "rb"), sheet_name="Sheet1" )
    for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 18]:
        building = pd.read_csv("./Daily-record/ID-" + str(i) + '-W.csv', \
            names=["i", "Time", "W_avg", "W_min", "W_max", "holiday", "count"], header=None)
        building = building.drop([0])
        building = building.reset_index()
        building['BuildingID'] = i
        building['Stair1'] = building_info_df.iloc[i]['Stair1']
        building['Stair2'] = building_info_df.iloc[i]['Stair2']
        building['Area'] = building_info_df.iloc[i]['Area']
        building['HVACType'] = building_info_df.iloc[i]['HVACType_num']       
        building.Time = pd.to_datetime(building.Time)
        building_1 = building[building['Time'] < pd.datetime(2016,2,29)]
        building_2 = building[building['Time'] >= pd.datetime(2016,3,1)]
        building = pd.concat([building_1, building_2])
        buildings.append(pd.merge(building, weather_df, on='Time'))
        # buildings.append(building)
    all_buildings = pd.concat(buildings)
    all_buildings.to_csv("all_buildings_daily_W.csv", index=False)

def create_daily_energy():
    df = pd.read_csv("6 train.csv")
    building_info_df = pd.read_excel(open("5 train_building_info.xlsx", "rb"), sheet_name="Sheet1" )
    weather_df = create_daily_weather()
    buildings = []
    for i in range(20):
        if i != 16:
            building = df[df['BuildingID'] == i]
            building['Stair1'] = building_info_df.iloc[i]['Stair1']
            building['Stair2'] = building_info_df.iloc[i]['Stair2']
            building['Area'] = building_info_df.iloc[i]['Area']
            building['HVACType'] = building_info_df.iloc[i]['HVACType_num']


            building_q = building[building['Type'] == 'Q']
            building_w = building[building['Type'] == 'W']
            # building = pd.concat([building_q, building_w])
            building_q = building_q.reset_index(drop=True)
            building_w = building_w.reset_index(drop=True)
            building_q['Record_Q'] = building_q['Record']
            building_q['Record_W'] = building_w['Record']
            building = building_q.drop(['Type', 'Record'], axis=1)
            # deal with anomaly numbers
            building[building[['Record_W']] < 0] = np.nan
            building[building[['Record_Q']] < 0] = np.nan
            building = building.fillna(method='ffill')

            if i == 0:
                building['Record_W'][building['Record_W'].idxmax()] = building['Record_W'][building['Record_W'].idxmax() - 1]
                building[building[['Record_Q']] > 1100] = np.nan
                building[building[['Record_W']] > 3000] = np.nan
                building = building.fillna(method='ffill')
            elif i == 1:
                building[building[['Record_Q']] > 1000] = np.nan
                building = building.fillna(method='ffill')
            elif i == 5:
                building[building[['Record_Q']] < 50] = np.nan
                building = building.fillna(method='ffill') 
            elif i == 10:
                building[building[['Record_Q']] < 50] = np.nan
                building = building.fillna(method='ffill')      

            building.Time = pd.to_datetime(building.Time).dt.strftime('%Y-%m-%d')
            building = building.groupby('Time').apply(f)
            building = building.reset_index()
            building.Time = pd.to_datetime(building.Time)

            building_1 = building[building['Time'] < pd.datetime(2016,2,29)]
            building_2 = building[building['Time'] >= pd.datetime(2016,3,1)]
            building = pd.concat([building_1, building_2])
            buildings.append(pd.merge(building, weather_df, on='Time'))
        # buildings.append(building)
    all_buildings = pd.concat(buildings)
    all_buildings.to_csv("all_buildings_daily.csv", index=False)

def f_weather(x):
    d = {}
    d['tem_avg'] = x['temp'].mean()
    d['tem_max'] = x['temp'].max()
    d['tem_min'] = x['temp'].min()

    d['tem_dew_avg'] = x['point_temp'].mean()
    d['tem_dew_max'] = x['point_temp'].max()
    d['tem_dew_min'] = x['point_temp'].min()

    d['rh_avg'] = x['humidity'].mean()
    d['rh_max'] = x['humidity'].max()
    d['rh_min'] = x['humidity'].min()

    d['pressure_avg'] = x['pa'].mean()
    d['pressure_max'] = x['pa'].max()
    d['pressure_min'] = x['pa'].min()

    d['vel_avg'] = x['wind speed'].mean()
    d['vel_max'] = x['wind speed'].max()
    d['vel_min'] = x['wind speed'].min()
    return pd.Series(d, index=['tem_avg', 'tem_max', 'tem_min', 'tem_dew_avg', 'tem_dew_max', 'tem_dew_min', \
     'rh_avg', 'rh_max', 'rh_min', 'pressure_avg', 'pressure_max', 'pressure_min', 'vel_avg', 'vel_max', 'vel_min'])

def f(x):
    d = {}
    d['Q_avg'] = x['Record_Q'].mean()
    d['Q_max'] = x['Record_Q'].max()
    d['Q_min'] = x['Record_Q'].min()
    d['Q_sum'] = x['Record_Q'].sum()
    
    d['W_avg'] = x['Record_W'].mean()
    d['W_max'] = x['Record_W'].max()
    d['W_min'] = x['Record_W'].min()
    d['W_sum'] = x['Record_W'].sum()
    
    d['BuildingID'] = x['BuildingID'][0]
    d['Stair1'] = x['Stair1'][0]
    d['Stair2'] = x['Stair2'][0]
    d['Area'] = x['Area'][0]
    d['HVACType'] = x['HVACType'][0]
    return pd.Series(d, index=['BuildingID','Q_avg', 'Q_max','Q_min','Q_sum','W_avg','W_max','W_min','W_sum', 'Stair1', 'Stair2', 'Area', 'HVACType'])

    
def create_regression_dataset():
    buildings = []
    df = pd.read_csv("6 train.csv")
    df.Time = pd.to_datetime(df.Time)

    weather_df1 = pd.read_excel(open('weather/shanghai weather_' + str(2015) + '.xlsx', 'rb'), sheet_name="Sheet2", 
                                names=["Time", "temp", "point_temp", "humidity", "pa", "wind speed"])
    weather_df2 = pd.read_excel(open('weather/shanghai weather_' + str(2016) + '.xlsx', 'rb'), sheet_name="Sheet2", 
                                names=["Time", "temp", "point_temp", "humidity", "pa", "wind speed"])
    weather_df3 = pd.read_excel(open('weather/shanghai weather_' + str(2017) + '.xlsx', 'rb'), sheet_name="Sheet2", 
                                names=["Time", "temp", "point_temp", "humidity", "pa", "wind speed"])
    # print(len(weather_df1), len(weather_df2), len(weather_df3))
    weather_df = pd.concat([weather_df1, weather_df2[:-1], weather_df3])
    weather_df.Time = pd.to_datetime(weather_df.Time)

    building_info_df = pd.read_excel(open("5 train_building_info.xlsx", "rb"), sheet_name="Sheet1" )
    for i in range(20):
        building = df[df['BuildingID'] == i]

        print(i, len(building))
        building['Stair1'] = building_info_df.iloc[i]['Stair1']
        building['Stair2'] = building_info_df.iloc[i]['Stair2']
        building['Area'] = building_info_df.iloc[i]['Area']
        building['HVACType'] = building_info_df.iloc[i]['HVACType_num']
        building_q = building[building['Type'] == 'Q']
        building_w = building[building['Type'] == 'W']
        # building = pd.concat([building_q, building_w])
        building_q = building_q.reset_index(drop=True)
        building_w = building_w.reset_index(drop=True)
        building_q['Record_Q'] = building_q['Record']
        building_q['Record_W'] = building_w['Record']
        building = building_q.drop(['Type', 'Record'], axis=1)
        # deal with anomaly numbers
        building[building[['Record_W']] < 0] = np.nan
        building[building[['Record_Q']] < 0] = np.nan
        building = building.fillna(method='ffill')

        if i == 0:
            building['Record_W'][building['Record_W'].idxmax()] = building['Record_W'][building['Record_W'].idxmax() - 1]
            building[building[['Record_Q']] > 1100] = np.nan
            building[building[['Record_W']] > 3000] = np.nan
            building = building.fillna(method='ffill')
        elif i == 1:
            building[building[['Record_Q']] > 1000] = np.nan
            building = building.fillna(method='ffill')
        elif i == 5:
            building[building[['Record_Q']] < 50] = np.nan
            building = building.fillna(method='ffill') 
        elif i == 10:
            building[building[['Record_Q']] < 50] = np.nan
            building = building.fillna(method='ffill') 





# building_df[building_df['Record_W'] > 50000].index
        # print(building)

        building_1 = building[building['Time'] < pd.datetime(2016,2,29)]
        building_2 = building[building['Time'] >= pd.datetime(2016,3,1)]
        building = pd.concat([building_1, building_2])
        buildings.append(pd.merge(building, weather_df, on='Time'))

    building_df = pd.concat(buildings)
    building_df.to_csv("all_buildings.csv", index=False)
# print(train_df)

def create_time_series_data():
    building_df = pd.read_csv("all_buildings.csv")
    building_df.Time = pd.to_datetime(building_df.Time).dt.strftime('%Y-%m-%d')
    today = pd.datetime(2015, 1, 1).dt.strftime('%Y-%m-%d')
    # while (today <= pd.datetime(2018, 1, 1).dt.strftime('%Y-%m-%d')):

    print(building_df.Time)

def daily_avg(energy):
    for i in range(20):
        try:
            df = pd.read_csv("ID-" + str(i) + "-" + energy + ".csv")
        except expression as identifier:
            pass
# create_regression_dataset()
# create_regression_dataset()
# create_daily_energy()
# create_daily_energy_Q()
# add_holiday()
create_testing_building_data()