# setwd()  
library(chron)
library(zoo)
library(stringr)
library(readxl)
library(timeDate)
library(grDevices);
ramp <- colorRamp(c("red", "white","blue"));
Sys.setlocale("LC_ALL","English")
Sys.setenv(TZ="Asia/Shanghai")
file_name = list.files('Hour-record/')
all_data = data.frame(row.names = NULL)
holiday = read_excel('holiday.xlsx')
holiday$holiday = as.character(holiday$holiday)



type_total = c('W','Q')
is_holiday_total = c(TRUE, FALSE)
ratio_median = matrix(data =  NA, nrow = 4, ncol = 24)
ratio_attir = matrix(data = NA, nrow = 4, ncol = 2)
d = 1
for(type in type_total){
  for(is_holiday in is_holiday_total){
    k=1
    ratio = matrix(nrow = length(file_name) * 4, ncol = 24) 
    for(i in file_name){
      if(str_detect(i, paste(type,'.csv',sep = ''))){
        data_i = read.csv(paste('Hour-record/',i,sep=''))
        data_i = data_i[!is.na(data_i$Record),]
        data_i$holiday = data_i$day %in% holiday$holiday
        weekend = !isWeekday(data_i$day)
        data_i$holiday = data_i$holiday | weekend
        data_i$Time = as.POSIXlt(data_i$Time, format = '%Y-%m-%d %H:%M:%S')
        data_i$hour = data_i$Time$hour
        data_i$month = months(data_i$Time)
        data_i$season = 'winter'
        data_i$season[data_i$month == 'February' |data_i$month == 'March'|data_i$month == 'April'] = 'spring'
        data_i$season[data_i$month == 'May' |data_i$month == 'June'|data_i$month == 'July'] = 'summer'
        data_i$season[data_i$month == 'August' |data_i$month == 'September'|data_i$month == 'October'] = 'autumn'
        for(j in c('winter', 'spring', 'summer', 'autumn')){
          data_j = data_i[data_i$season == j & data_i$holiday == is_holiday,]
          ratio[k,] = tapply(data_j$Record, data_j$hour, mean, na.rm = T)
          k = k+1
        }
      }
    }
    
    for(i in 1:nrow(ratio)){
      ratio[i,] = ratio[i,]/max(ratio[i,])
    }
    ratio_median[d,] = apply(ratio, 2, median, na.rm=T)
    ratio_attir[d,] = c(type, is_holiday)
    d = d+1
    
    for(i in 1:4){
      season = c('winter.jpg', 'spring.jpg', 'summer.jpg', 'autumn.jpg')
      jpeg(paste('Result_figure/Result',type,is_holiday, season[i], sep = '-'))
      plot(ratio[i,], ylim = c(0,1), col = rgb(23,45,255, maxColorValue = 255))
      k = i+4
      j=2
      while(k<=nrow(ratio)){
        lines(ratio[k,])
        k = k+4
        j=j+1
      }
      dev.off()
    }
  }
}

result = cbind(ratio_attir, ratio_median)
write.csv(result, 'hourly_median.csv', row.names = FALSE,col.names = FALSE)
