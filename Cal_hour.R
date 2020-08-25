# setwd()  
#
#
# 可能需要你这这里把文件夹设为工作路径,如：C:\Users\xilei\OneDrive - Macquarie University\BuildingEnergy
library(chron)
library(zoo)
library(stringr)
library(readxl)
library(timeDate)
library(grDevices)
rm(list = ls())
ramp <- colorRamp(c("red", "white","blue"));
Sys.setlocale("LC_ALL","English")
Sys.setenv(TZ="Asia/Shanghai")
ratio = read.csv('hourly_median.csv')
W_True = as.numeric(ratio[1,3:26])/mean(as.numeric(ratio[1,3:26]))
W_False = as.numeric(ratio[2,3:26])/mean(as.numeric(ratio[2,3:26]))
Q_True = as.numeric(ratio[3,3:26])/mean(as.numeric(ratio[3,3:26]))
Q_False = as.numeric(ratio[4,3:26])/mean(as.numeric(ratio[4,3:26]))
holiday = read_excel('holiday.xlsx')
holiday$holiday = as.character(holiday$holiday)

Q_daily = read.csv('output_Q.csv')
Q_daily$prediction[Q_daily$prediction == 0] = NA
if(is.na(Q_daily$prediction[1])){
  Q_daily$prediction[1] = Q_daily$prediction[min(Qhich(is.na(Q_daily$prediction)==FALSE))]
}
if(is.na(Q_daily$prediction[365])){
  Q_daily$prediction[1] = Q_daily$prediction[min(Qhich(is.na(Q_daily$prediction)==FALSE))]
}
Q_daily$prediction = na.approx(Q_daily$prediction)
Q_daily$holiday = Q_daily$Time %in% holiday$holiday
Q_daily$Time = as.POSIXct(Q_daily$Time, format = '%Y-%m-%d')
Q_daily$weekend = isWeekend(cut(Q_daily$Time, breaks = 'day'))
Q_daily$is_holiday = Q_daily$holiday | Q_daily$weekend
Q_daily = Q_daily[,c(1,2,5)]

W_daily = read.csv('output_W.csv')
W_daily$prediction[W_daily$prediction == 0] = NA
if(is.na(W_daily$prediction[1])){
  W_daily$prediction[1] = W_daily$prediction[min(which(is.na(W_daily$prediction)==FALSE))]
}
if(is.na(W_daily$prediction[365])){
  W_daily$prediction[365] = W_daily$prediction[min(which(is.na(W_daily$prediction)==FALSE))]
}
W_daily$prediction = na.approx(W_daily$prediction)
W_daily$holiday = W_daily$Time %in% holiday$holiday
W_daily$Time = as.POSIXct(W_daily$Time, format = '%Y-%m-%d')
W_daily$weekend = isWeekend(cut(W_daily$Time,breaks = 'day'))
W_daily$is_holiday = W_daily$holiday | W_daily$weekend
W_daily = W_daily[,c(1,2,5)]

beg_T = as.POSIXct('2017-1-1 00:00:00', format = '%Y-%m-%d %H:%M:%S')
end_T = as.POSIXct('2017-12-31 23:00:00', format = '%Y-%m-%d %H:%M:%S')
Q_hour = data.frame(time = seq(beg_T, end_T, by = 3600), record = NA)
for(i in 1:365){
  if(Q_daily$is_holiday[i]){
    Q_hour$record[((i-1)*24 + 1) : (i*24)] = Q_daily$prediction[i] * Q_True
  }
  else{
    Q_hour$record[((i-1)*24 + 1) : (i*24)] = Q_daily$prediction[i] * Q_False
  }
}

W_hour = data.frame(time = seq(beg_T, end_T, by = 3600), record = NA)
W_hour = data.frame(time = seq(beg_T, end_T, by = 3600), record = NA)
for(i in 1:365){
  if(W_daily$is_holiday[i]){
    W_hour$record[((i-1)*24 + 1) : (i*24)] = W_daily$prediction[i] * W_True
  }
  else{
    W_hour$record[((i-1)*24 + 1) : (i*24)] = W_daily$prediction[i] * W_False
  }
}


result = read.csv('1 result.csv')
result$Time = as.POSIXct(result$Time, format = '%Y-%m-%d %H:%M:%S')
for(i in 1:nrow(result)){
  if(result$Type[i] == 'Q'){
    result$Record[i] = Q_hour$record[which(Q_hour$time == result$Time[i])]
  }
  if(result$Type[i] == 'W'){
    result$Record[i] = W_hour$record[which(W_hour$time == result$Time[i])]
  }
}
write.csv(result, file = 'ED01E10020+result+2.csv', row.names = FALSE)
