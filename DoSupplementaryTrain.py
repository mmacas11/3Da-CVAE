#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


# In[2]:


###time computation
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# In[3]:


losses = []


# In[4]:


#loss function
def handleLoss(loss):
        global losses
        losses+=[loss]


# In[5]:


class LossHistory( tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        handleLoss(logs.get('loss'))


# In[6]:


def plotimages(data,numc):
    images = np.array(data)
    images = images.reshape(-1,int(np.sqrt(data.shape[1])),int(np.sqrt(data.shape[1])))
    print("Images shape = ",images.shape,"\nLabels shape = ")
    print(type(images))    
    plt.figure(1 , figsize = (19 , 10))
    n = 0 
    for i in range(numc):
        n += 1 
        r = np.random.randint(0 , images.shape[0] , 1)
        
        plt.subplot(int(np.sqrt(numc)), int(np.sqrt(numc)) , n)
        plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
        plt.imshow(images[r[0]])
        
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[7]:


def saverecord (listinput,name):
    #save the list in output.bin file
    with open(name, "wb") as output:
        pickle.dump(listinput, output)


# In[8]:


def savetrainingprocess (history,model_name):
        #Evaluation --label=model_name
        #loss 
        plt.plot(history['loss'])
        plt.title(model_name + 'loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right');        
        plt.savefig(model_name+ ' loss')
        plt.clf() 


# In[9]:


def noisy_data (data,stddev,noise_factor):
    data_noise = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return data_noise


# In[10]:


def save_all_detail_model (model, name_model,times,history):
    timetraining = []
    hytories = []
    if name_model == 'vae':
        model.encoder.save('vae_encoder.tf')
        model.decoder.save('vae_decoder.tf')
    else:
        model.save('model_'+name_model+'.tf')
    timetraining.append(times)
    hytories.append(history)
    saverecord (hytories,'hytories_'+ name_model +'.bin')
    saverecord (timetraining,'timetraining_'+ name_model +'.bin')
    savetrainingprocess (history,name_model)


# In[11]:


def load_model(name):
    new_model = tf.keras.models.load_model(name)
    return new_model

