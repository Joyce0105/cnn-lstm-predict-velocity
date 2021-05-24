import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import Variable

from dataset_cnn import getData,testData
from cnn import CNN
from cnn_iter import train

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

testspeed_min,testspeed_max,test_loader = testData(net_ornot=1)
_,_,checkpoint = train(test_ornot=1,netornot=1,iter=420)

model_test = CNN()
model_test.load_state_dict(checkpoint['state_dict'])
for x,y in test_loader:
    pred = model_test(x)
pred = np.array(pred.detach())*(testspeed_max-testspeed_min)+testspeed_min
y = np.array(y.view(y.size(0),-1).detach())*(testspeed_max-testspeed_min)+testspeed_min
z = pred-y
MSE = np.mean(np.power(z,2))
MAE = np.mean(np.abs(z))
MAPE = np.mean(np.abs(z)/y)

print('MSE:',MSE)
print('RMSE',np.sqrt(MSE))
print('MAE:',MAE)
print('MAPE:',MAPE)

pred = pred.reshape(288,184,3)
y = pred.reshape(288,184,3)


plt.figure()
for i in range(6):
    data1 = pred[48*i:48*i+48]
    data2 = data1[0]
    data11 = y[48*i:48*i+48]
    data22 = data11[0]
    for j in np.arange(1,48):
        data2 = np.hstack((data2,data1[j]))
        data22 = np.hstack((data22,data11[j]))
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







