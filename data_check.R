# setwd()  
library(chron)
library(zoo)
Sys.setlocale("LC_ALL","Chinese")
Sys.setenv(TZ="Asia/Shanghai")
data = read.csv('6 train.csv')
data$Time = as.POSIXct(data$Time, format = '%Y-%m-%d %H:%M:%S')
data$day = cut(data$Time, breaks='day')
day_seq = unique(data$day)
beg_T = as.POSIXct('2015-1-1 0:0:0', format = '%Y-%m-%d %H:%M:%S')
end_T = as.POSIXct('2017-12-31 23:0:0', format = '%Y-%m-%d %H:%M:%S')
time_seq = seq(from = beg_T, to = end_T, by='hour')
ratio_qw = rep(0,20)
data$Record[data$Record<=0] = NA
for(i in 0:19){
 record_q = data$Record[data$BuildingID==i & data$Type=='Q']
 record_w = data$Record[data$BuildingID==i & data$Type=='W']
 ratio_qw[i+1] = median(record_q/record_w, na.rm = T)
}
plot(ratio_qw)
data$range = cut(data$Record,c(seq(0,10000,1000),Inf))
table(data$range)
data$Record[data$Record > 4000] = NA
print(which(ratio_qw>10)-1)


#remove median q/w>10
for(i in c(0,1,3:10,14:16,18,19)){
  for(j in c('Q','W')){
    if(i == 6 & j=='Q'){
      next
    }
    data_i = data[data$BuildingID == i & data$Type == j,]
    data_full = data.frame(Time = time_seq, BuildingID = NA, Type = NA, Record = NA, day = NA, count = 1)
    data_full$BuildingID = data_i[match(data_full$Time, data_i$Time),2]
    data_full$Type = data_i[match(data_full$Time, data_i$Time),3]
    data_full$Record = data_i[match(data_full$Time, data_i$Time),4]
    data_full$day = data_i[match(data_full$Time, data_i$Time),5]
    data_full = data_full[!is.na(data_full$BuildingID),]
    data_full$ratio = NA
    data_full$ratio[2:(nrow(data_full)-1)] = data_full$Record[2:(nrow(data_full)-1)]/((data_full$Record[1:(nrow(data_full)-2)]+
                                                                                         data_full$Record[3:(nrow(data_full))])/2)
    outliner = boxplot(data_full$ratio)
    lim = outliner$stats[5]*3
    data_full$Record[data_full$ratio>lim]=NA
#    if(table(is.na(data_full$Record))[1] == nrow(data_full)){
#      next
#    }
    data_full$Record = na.approx(data_full$Record)
    if(i == 19){
      ET = as.POSIXct('2016-12-31 23:00:00', format = '%Y-%m-%d %H:%M:%S')
      data_full = data_full[data_full$Time<=ET,]
    }
    file_name = paste('ID',i,j,sep = '-')
    file_name = paste('Hour-record/',file_name,'.csv',sep = '')
  
    write.csv(data_full,file_name)    

  }
}
