#!/usr/bin/env python
# coding: utf-8

# In[1102]:


import collections
import numpy as np
import pandas as pd


# In[1103]:


def load_data (data_to_pathX, data_to_pathY):
    """
    #Load data
    """
    print ("Laoding the sensor data")
    data_X = pd.read_csv(data_to_pathX,engine='python',sep=",",header=0) 
    data_Y = pd.read_csv(data_to_pathY,engine='python',sep=",",header=0)
    return data_X,data_Y #yield


# In[1104]:


def isMultiple(num,  check_with):
    return num % check_with == 0


# In[1105]:


def get_length_data (datalen,batch_size,val_percent,timestep):
    #length of the entire normal dataset
    lengtht = datalen
    lengtht *= 1-val_percent
    for i in range(int(lengtht),0,-1):
        if (isMultiple(i, batch_size) == True):
            if (isMultiple(i, timestep) == True):
                lengtht = i
                print(lengtht)
                break
    return (lengtht)


# In[1106]:


def split_data(dataX,datay,batch_size,val_percent,timestep):
    datalen = len(dataX)
    lengtht = get_length_data (datalen,batch_size,val_percent,timestep)
    Xtrain = dataX.iloc[0:lengtht]
    Xval = dataX.iloc[lengtht:]
    ytrain = datay.iloc[0:lengtht]
    yval = datay.iloc[lengtht:]
    return(Xtrain,Xval,ytrain,yval)


# In[1107]:


def add_timestep_rnn(data,timestep):
    samples = list()
    for i in range(0,(len(data)-timestep)+1,timestep):
        sample = data[i:i+timestep]
        samples.append(sample)
    return (np.array(samples, dtype=np.float32))


# In[1108]:


def reshape_rnn(dXt,dXv,dyt,dyv,timestep):
    numft = dXt.shape[1]
    xypad = int(np.sqrt(numft))
    dXt_tsp = add_timestep_rnn(dXt,timestep)
    dXv_tsp = add_timestep_rnn(dXv,timestep)
    dyt_tsp = dyt.values
    dyv_tsp = dyv.iloc[0:(dXv_tsp.shape[0]*dXv_tsp.shape[1])].values
    dXt_tsp = dXt_tsp.reshape(dXt_tsp.shape[0], dXt_tsp.shape[1], xypad, xypad, 1)
    dXv_tsp = dXv_tsp.reshape(dXv_tsp.shape[0], dXv_tsp.shape[1], xypad, xypad, 1)
    dyt_tsp = dyt_tsp.reshape(dXt_tsp.shape[0], dXt_tsp.shape[1], 1)
    dyv_tsp = dyv_tsp.reshape(dXv_tsp.shape[0], dXv_tsp.shape[1], 1)
    return(dXt_tsp,dXv_tsp,dyt_tsp,dyv_tsp)


# In[1109]:


def reshape_cnn(dXt,dXv,dyt,dyv):
    numft = dXt.shape[1]
    xypad = int(np.sqrt(numft))
    dXt_R = dXt.values.reshape(dXt.shape[0], xypad, xypad, 1)
    dXv_R = dXv.values.reshape(dXv.shape[0], xypad, xypad, 1)
    dyt_R = dyt
    dyv_R = dyv
    return(dXt_R,dXv_R,dyt_R,dyv_R)


# In[1110]:


def load_split_data (idata_pathX,idata_pathY,ibatchsize,ipercentval,itimestep,typemodel):
    data_pathX = idata_pathX
    data_pathY = idata_pathY
    x,y=load_data(data_pathX,data_pathY)
    batchsize = ibatchsize
    percentval = ipercentval
    timestep = itimestep
    xtrain,xval,ytrain,yval = split_data(x,y,batchsize,percentval,timestep)
    dXt_tsp,dXv_tsp,dyt_tsp,dyv_tsp = reshape_rnn(xtrain,xval,ytrain,yval,timestep)
    dXt_R,dXv_R,dyt_R,dyv_R = reshape_cnn(xtrain,xval,ytrain,yval)    
    if (typemodel=='ffnn'):
        dataop = collections.namedtuple('ffnn', 'Xtran ytrain Xval yVal')
        model_train_dataop = dataop(Xtran=dXt_R, ytrain=dyt_R,Xval=dXv_R, yVal=dyv_R)        
    elif(typemodel=='rnn'):
        dataop = collections.namedtuple('rnn', 'Xtran ytrain Xval yVal')
        model_train_dataop = dataop(Xtran=dXt_tsp, ytrain=dyt_tsp,Xval=dXv_tsp, yVal=dyv_tsp)
    else:
        dataop = collections.namedtuple('traditional', 'Xtran ytrain Xval yVal')
        model_train_dataop = dataop(Xtran=xtrain, ytrain=ytrain,Xval=xval, yVal=yval)    
    c_data_train_model = collections.namedtuple('c_data_train_model', 'model_train_data')        
    data_train_models_s = c_data_train_model(model_train_data=model_train_dataop) 
    yield data_train_models_s

