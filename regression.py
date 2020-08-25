import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import numpy as np
from utils import cv_rmse




if __name__ == "__main__":    
    df = pd.read_csv("all_buildings.csv")

    df = df.drop(["Time", "BuildingID",], axis=1)
    scale_feature = ['Stair1','Stair2','Area','temp','point_temp','humidity','pa','wind speed',  "Record_W", "Record_Q"]


    scaler = MinMaxScaler()
    scaler.fit(df[scale_feature])
    df[scale_feature] = scaler.transform(df[scale_feature])
    label_w = df["Record_W"]
    label_q = df["Record_Q"]
    df = df.drop(["Record_W", "Record_Q"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, label_w, test_size=0.2, random_state=42)
    # reg = LinearRegression().fit(X_train, y_train)
    # print(reg.coef_)
    # y_pred = reg.predict(X_test)
    # # print(mean_squared_error(y_test, y_pred))
    # print(cv_rmse(y_test, y_pred))

    svm_reg = RandomForestRegressor()
    svm_reg.fit(X_train, y_train)
    y_pred = svm_reg.predict(X_test)
    # print(mean_squared_error(y_test, y_pred))
    print(cv_rmse(y_test, y_pred))
