from torch.autograd import Variable
import torch.nn as nn
import torch
from dataset_cnn import getData
from cnn import CNN

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

def train(iter=500,test_ornot=0):

    model = CNN()
    model_val = model

    speed_min,speed_max,train_loader,val_loader,all_train_loader = getData()

    if test_ornot == 1:
        train_loader = all_train_loader

    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    criterion = nn.MSELoss()

    Loss_train = []
    Loss_val = []

    for i in range(iter):
        Loss = 0
        for x,y in train_loader:
            pred = model(Variable(x))
            loss = criterion(pred, y.view(y.size(0),-1))
            Loss = Loss+loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        Loss_train.append(Loss)

        if test_ornot == 0:
            if (i+1)%10==0:
                torch.save({'state_dict': model.state_dict()}, 'C:/Users/Joyce/Desktop/毕业论文/数院代码/math2/cnn.pkl')
                checkpoint = torch.load('C:/Users/Joyce/Desktop/毕业论文/数院代码/math2/cnn.pkl')
                model_val.load_state_dict(checkpoint['state_dict'])
                for x,y in val_loader:
                    pred = model_val(x)
                    loss = criterion(pred, y.view(y.size(0),-1))
                Loss_val.append(loss.item())

        if i == iter-1:
            torch.save({'state_dict': model.state_dict()}, 'C:/Users/Joyce/Desktop/毕业论文/数院代码/math2/cnn.pkl')
            checkpoint = torch.load('C:/Users/Joyce/Desktop/毕业论文/数院代码/math2/cnn.pkl')

    return Loss_train,Loss_val,checkpoint

if __name__ == '__main__':

    Loss_train,Loss_val,_ = train()
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False

    print(np.argmin(Loss_val))
    #print(Loss_train)
    #print(Loss_val)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(Loss_train)
    plt.title('训练历史')
    plt.xlabel('迭代次数')
    plt.ylabel('标准化训练集的MSE')

    plt.subplot(1,2,2)
    plt.plot(Loss_val,c='orange')
    #plt.scatter(np.argmin(Loss_val),np.min(Loss_val),c='red')
    plt.title('验证历史')
    plt.xlabel('验证次数')
    plt.ylabel('标准化验证集的MSE')

    plt.show()
