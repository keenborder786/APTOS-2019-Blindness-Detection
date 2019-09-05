# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:42:08 2019

@author: MMOHTASHIM
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def Generate_Data_train():
  ###
  #Description
  #This function will take the images from train folder and covert them to an np.array of following shape=(number of images,224,244,3)
  #Please note each instance of image is of shape 224,224,3 which means that image is colored of dims 224x224. I have create a simple iterartive
  #algorithm which will read the id from associated train csv files and then read the this id from image folde. After reading I will make use of cv2 resize and then append the image pixel
  #data to X_train.
  
#   Returns-Nothing, Need to run only one time to save the data(X_train)
  ####
  
  os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Blindness-Detection")
  X_test=[]
  df_labels=pd.read_csv("test.csv")
  ids=df_labels["id_code"].values.reshape(-1,)
  print(ids)
  for id in tqdm(ids):
    ##try and except to capture for an ids which have errors
    try:
      img_dir=r"C:\Users\MMOHTASHIM\Anaconda3\libs\Blindness-Detection\test_images"
      ##reading the image and then resizing it,RGB scale
      img=cv2.imread(img_dir+"/"+f"{id}.png",1)
      img=cv2.resize(img, (224, 224))

      X_test.append(img)
    except:
      print(id)
  ##declaring it as array
  X_test=np.array(X_test)
  
  np.save("X_test.npy",X_test)
def one_hot_encode_y():
    ###
    #Description
    #This function will one hot encode y to make labels more efficent in regard to the neural network
    
    #Returns-Nothing, Need to run only one time to save the data(y_train)
    ###
    df_labels=pd.read_csv("train.csv")
    labels=df_labels["diagnosis"].values.reshape(1,-1)
    unique_values=list(set(labels[0]))
    y_train=[]
    for label in tqdm(labels[0]):
        #one-hot-encoding
        one_hot_encode=list(np.zeros(len(unique_values)))
        index_label=unique_values.index(label)
        one_hot_encode[index_label]=1
        y_train.append(one_hot_encode)
    np.array(y_train)
    np.save("y_train.npy",y_train)
def kaggle_submission():
    y_test=np.load("y_test.npy")
    df=pd.DataFrame()
    df["Categories"]=y_test
    df.to_csv("InceptionResNetV2-noclass.csv")
def my_result_accuracy():
    df=pd.read_csv("Benchmark.csv")
    count=0
    for i in range(len(df)):
        if df.iloc[i,1]==df.iloc[i,2]:
            count+=1
    acc=count/len(df)
    print(f"The accuracy is {acc}")
        
        
        
    
    
  
if '__main__' == __name__:
    kaggle_submission()
    my_result_accuracy()