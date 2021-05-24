import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

def getData(batchSize=512):

    data1 = pd.read_csv('C:/Users/Joyce/Desktop/math3.csv')
    data2 = pd.read_csv('C:/Users/Joyce/Desktop/math4.csv')
    data = data1.append(data2)
    data = data.sort_values(['road_id','day_id','time_id'])
        

    data.index = range(data.shape[0])
    speed_max = np.max(data['speed'])
    speed_min = np.min(data['speed'])
   
    df=pd.DataFrame()
    for i in range(data['road_id'].nunique()):
        df[i+1] = (data[data['road_id']==i+1]['speed'].values - speed_min)/(speed_max-speed_min)
    num = data[data['road_id']==1]['speed'].count()

    seq_x = 12
    seq_y = 3
    X = []
    Y=[]

    for i in np.arange(0,num-seq_x,seq_y):
        X.append(np.array(df.iloc[i:i+seq_x,].values,dtype=np.float32))
        Y.append(np.array(df.iloc[i+seq_x:i+seq_x+seq_y,].values,dtype=np.float32))

    X = torch.tensor(X)
    #print(Y)
    Y = torch.tensor(Y).view(len(Y),-1)
    #print(Y.shape)

    train_size = len(Y)-6*48

    train_set = TensorDataset(X[:train_size],Y[:train_size])
    train_loader = DataLoader(train_set,batch_size=batchSize,shuffle=True)

    val_set = TensorDataset(X[train_size:],Y[train_size:])
    val_loader = DataLoader(val_set,batch_size=6*48)
    
    all_train_set = TensorDataset(X,Y)
    all_train_loader = DataLoader(all_train_set,batch_size=batchSize,shuffle=True)

    return speed_min,speed_max,train_loader,val_loader,all_train_loader

def testData():

    data = pd.read_csv('C:/Users/Joyce/Desktop/test3.csv')

    data.index = range(data.shape[0])
    speed_max = np.max(data['speed'])
    speed_min = np.min(data['speed'])
    

    df=pd.DataFrame()
    for i in range(data['road_id'].nunique()):
        df[i+1] = (data[data['road_id']==i+1]['speed'].values - speed_min)/(speed_max-speed_min)
    #df.apply(lambda x : (x-speed_min)/(speed_max - speed_min))

    num = data[data['road_id']==1]['speed'].count()

    
    seq_x = 12
    seq_y = 3
    X = []
    Y=[]

    for i in np.arange(0,num-seq_x,seq_y):
        X.append(np.array(df.iloc[i:i+seq_x,].values,dtype=np.float32))
        Y.append(np.array(df.iloc[i+seq_x:i+seq_x+seq_y,].values,dtype=np.float32))

    X = torch.tensor(X)
    Y = torch.tensor(Y).view(len(Y),-1)

    test_set = TensorDataset(X,Y)
    test_loader = DataLoader(test_set,batch_size=len(Y))

    return speed_min,speed_max,test_loader
