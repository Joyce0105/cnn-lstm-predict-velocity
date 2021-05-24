import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch


def getData(batchSize=512):
    data1 = pd.read_csv('C:/Users/Joyce/Desktop/math3.csv')
    data2 = pd.read_csv('C:/Users/Joyce/Desktop/math4.csv')
    data = data1.append(data2)
    data = data.sort_values(['road_id','day_id','time_id'])
    
    data.index = range(data.shape[0])
    speed_max = np.max(data['speed'])
    speed_min = np.min(data['speed'])

    df=pd.DataFrame()
    df['road_id'] = data['road_id']
    df['speed'] = (data['speed']-speed_min)/(speed_max - speed_min)

    num = df[df['road_id']==1]['speed'].count()

    seq_x = 12
    seq_y = 3
    X = []
    Y=[]

    x = []
    for i in range(df['road_id'].nunique()):
        x.append(np.array(df[df['road_id']==i+1]['speed'].values,dtype=np.float32))
    x = np.matrix(x)

    for i in np.arange(0,num-seq_x,seq_y):
        X.append(np.array(x[:,i:i+seq_x],dtype=np.float32))
        Y.append(np.array(x[:,i+seq_x:i+seq_x+3],dtype=np.float32))

    Y = torch.tensor(Y)
    X = torch.tensor(X)
    X = X.unsqueeze(1) # channel维
    Y = Y.unsqueeze(1)

    train_size = len(X)-288
    train_X = X[:train_size]
    train_Y = Y[:train_size]
    val_X = X[train_size:]
    val_Y = Y[train_size:]

    train_set = TensorDataset(train_X,train_Y)
    val_set = TensorDataset(val_X,val_Y)

    train_loader = DataLoader(train_set,batch_size=batchSize,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=len(val_Y))

    all_train_loader = DataLoader(TensorDataset(X,Y),batch_size=batchSize,shuffle=True)

    return speed_min,speed_max,train_loader,val_loader,all_train_loader

def testData():
    
    data = pd.read_csv('C:/Users/Joyce/Desktop/test3.csv')

    data.index = range(data.shape[0])
    speed_max = np.max(data['speed'])
    speed_min = np.min(data['speed'])

    df=pd.DataFrame()
    df['road_id'] = data['road_id']
    df['speed'] = (data['speed']-speed_min)/(speed_max - speed_min)

    num = df[df['road_id']==1]['speed'].count()

    seq_x = 12
    seq_y = 3
    X = []
    Y=[]

    x = []
    for i in range(df['road_id'].nunique()):
        x.append(np.array(df[df['road_id']==i+1]['speed'].values,dtype=np.float32))
    x = np.matrix(x)

    for i in np.arange(0,num-seq_x,seq_y):
        X.append(np.array(x[:,i:i+seq_x],dtype=np.float32))
        Y.append(np.array(x[:,i+seq_x:i+seq_x+3],dtype=np.float32))

    Y = torch.tensor(Y)
    X = torch.tensor(X)
    X = X.unsqueeze(1) # channel维
    Y = Y.unsqueeze(1)

    test_set = TensorDataset(X,Y)
    test_loader = DataLoader(test_set,batch_size=len(Y))

    return speed_min,speed_max,test_loader