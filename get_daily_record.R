# setwd()  
library(chron)
library(readxl)
Sys.setlocale("LC_ALL","Chinese")
Sys.setenv(TZ="Asia/Shanghai")

for(i in c(0,1,3:10,14:16,18,19)){
  for(j in c('Q','W')){
    if(i == 6 & j=='Q'){
      next
    }

    file_name = paste('ID',i,j,sep = '-')
    file_name = paste('Hour-record/',file_name,'.csv',sep = '')
    data_full = read.csv(file = file_name)
    data_full$Time = as.POSIXlt(data_full$Time, format = '%Y-%m-%d %H:%M:%S')
    day_seq = unique(cut(data_full$Time, breaks = 'day'))

    result = data.frame(Day = day_seq, avg = NA,  min = NA, max = NA, holiday = NA, count = NA)
    result$Day = as.POSIXct(result$Day, format = '%Y-%m-%d')
    result$avg =  tapply(data_full$Record, data_full$day, mean, na.rm = T)
    result$min =  tapply(data_full$Record, data_full$day, min, na.rm = T)
    result$max =  tapply(data_full$Record, data_full$day, max, na.rm = T)
    result$count =  tapply(data_full$count, data_full$day, sum, na.rm = T)
    result$holiday = as.numeric(is.weekend(result$Day))
    file_name = paste('ID',i,j,sep = '-')
    file_name = paste('Daily-record/',file_name,'.csv', sep = '')
    
    if(i == 19 & j=='W'){
      beginT = as.POSIXct('2015-1-1 0:0:0', format = '%Y-%m-%d %H:%M:%S')
      endT = as.POSIXct('2016-12-31 0:0:0', format = '%Y-%m-%d %H:%M:%S')
      result[result$Time>=beginT & result$Time<= endT,2:4] = NA
    }
    
    if(i == 19 & j=='Q'){
      beginT = as.POSIXct('2016-1-1 0:0:0', format = '%Y-%m-%d %H:%M:%S')
      endT = as.POSIXct('2016-12-31 0:0:0', format = '%Y-%m-%d %H:%M:%S')
      result[result$Time>=beginT & result$Time<= endT,2:4] = NA
    }
    
    if(i == 1){
      month_id = months(result$Day)
      index = month_id=='一月' | month_id=='十一月' | month_id=='十二月'
      result[index,2:4] = NA
    }
    
    if(i == 2){
      month_id = months(result$Day)
      index = month_id=='一月' | month_id=='十一月' | month_id=='十二月'
      result[index,2:4] = NA
    }
    
    holiday = read_excel('holiday.xlsx', sheet = 1)
    holiday$holiday = as.character(holiday$holiday)
    holiday$holiday = as.POSIXct(holiday$holiday , format = '%Y-%m-%d')
    holiday_index = result$Day %in% holiday$holiday
    
    result$holiday[holiday_index] = 1
    
    change = (result$avg[2:nrow(result$avg)] - result$avg[1:(nrow(result$avg)-1)])/result$avg[2:nrow(result$avg)]
    change = which(change < 0.01)
    #result[change+1,2:4] = NA
      
    
    
    file_name = paste('ID',i,j,sep = '-')
    file_name = paste('Daily-record/',file_name,'.csv',sep = '')
    write.csv(result, file_name)
  }
}
