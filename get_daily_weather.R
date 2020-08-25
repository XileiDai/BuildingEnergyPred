# setwd()  
library(chron)
library(xlsx)
library(readxl)
Sys.setlocale("LC_ALL","Chinese")
Sys.setenv(TZ="Asia/Shanghai")
file_name = c('shanghai weather_2015.xlsx', 'shanghai weather_2016.xlsx', 'shanghai weather_2017.xlsx')
for(i in 1:3){
  path = paste('weather', file_name[i], sep = '/')
  data = read_excel(path, sheet = 1)
  data$time = as.POSIXct(data$time, format = '%Y-%m-%d %H:%M:%S')
  colnames(data) = c('time', 'tem', 'tem_dew', 'rh', 'pressure', 'vel')
  data$day = cut(data$time, breaks = 'day')
  tem_avg = tapply(data$tem, data$day, mean, na.rm = T)
  tem_min = tapply(data$tem, data$day, min, na.rm = T)
  tem_max = tapply(data$tem, data$day, max, na.rm = T)
  
  tem_dew_avg = tapply(data$tem_dew, data$day, mean, na.rm = T)
  tem_dew_min = tapply(data$tem_dew, data$day, min, na.rm = T)
  tem_dew_max = tapply(data$tem_dew, data$day, max, na.rm = T)
  
  rh_avg = tapply(data$rh, data$day, mean, na.rm = T)
  rh_min = tapply(data$rh, data$day, min, na.rm = T)
  rh_max = tapply(data$rh, data$day, max, na.rm = T)
  
  pressure_avg = tapply(data$pressure, data$day, mean, na.rm = T)
  pressure_min = tapply(data$pressure, data$day, min, na.rm = T)
  pressure_max = tapply(data$pressure, data$day, max, na.rm = T)
  
  vel_avg = tapply(data$vel, data$day, mean, na.rm = T)
  vel_min = tapply(data$vel, data$day, min, na.rm = T)
  vel_max = tapply(data$vel, data$day, max, na.rm = T)
  
  result = data.frame(tem_avg = tem_avg, tem_min = tem_min, tem_max = tem_max,
                      tem_dew_avg = tem_dew_avg, tem_dew_min = tem_dew_min, tem_dew_max = tem_dew_max,
                      rh_avg = rh_avg, rh_min = rh_min, rh_max = rh_max,
                      pressure_avg = pressure_avg, pressure_min = pressure_min, pressure_max = pressure_max,
                      vel_avg = vel_avg, vel_min = vel_min, vel_max = vel_max)
  
  save_path = paste('Daily-weather/weather-',i+2014,'.csv', sep = '')
  write.csv(result, save_path)
}
