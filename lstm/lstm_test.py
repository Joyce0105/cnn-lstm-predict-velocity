import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset

from dataset_lstm import getData,testData
from LSTM import Lstm
from lstm_iter import train

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False


testspeed_min,testspeed_max,test_loader= testData(net_ornot=1)
model_test = Lstm()

_,_,checkpoint = train(test_ornot=1,iter=700)
model_test.load_state_dict(checkpoint['state_dict'])

for x,y in test_loader:
    pred = model_test(x)
#print(pred.size(1))
pred = np.array(pred.detach())*(testspeed_max-testspeed_min)+testspeed_min
y = np.array(y.detach())*(testspeed_max-testspeed_min)+testspeed_min
z = pred-y
MSE = np.mean(np.power(z,2))
MAE = np.mean(np.abs(z))
MAPE = np.mean(np.abs(z)/y)

print('MSE:',MSE)
print('RMSE',np.sqrt(MSE))
print('MAE:',MAE)
print('MAPE:',MAPE)

pred = pred.reshape(288,3,184)
y = y.reshape(288,3,184)


plt.figure()
for i in range(6):
    data1 = pred[48*i:48*i+48]
    data2 = data1[0].T
    data11 = y[48*i:48*i+48]
    data22 = data11[0].T
    for j in np.arange(1,48):
        data2 = np.hstack((data2,data1[j].T))
        data22 = np.hstack((data22,data11[j].T))
    if i <3:
        plt.subplot(2,3,i+1)
        sns.heatmap(data2,cmap='OrRd_r')
        if i == 0:
            plt.ylabel('路段编号')
        plt.xlabel('时间编号')
        plt.title('第{}天预测值'.format(i+1))
        plt.subplot(2,3,i+3+1)
        sns.heatmap(data22,cmap='GnBu_r')
        if i==0:
            plt.ylabel('路段编号')
        plt.xlabel('时间编号')
        plt.title('第{}天真实值'.format(i+1))
    if i ==2:
        plt.show()
        plt.figure()
    if i >=3:
        plt.subplot(2,3,i-3+1)
        sns.heatmap(data2,cmap='OrRd_r')
        if i == 3:
            plt.ylabel('路段编号')
        plt.xlabel('时间编号')
        plt.title('第{}天预测值'.format(i+1))
        plt.subplot(2,3,i+1)
        sns.heatmap(data22,cmap='GnBu_r')
        if i==3:
            plt.ylabel('路段编号')
        plt.xlabel('时间编号')
        plt.title('第{}天真实值'.format(i+1))
plt.show()



