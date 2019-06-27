# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:08:38 2019
    
@author: Tony.George
"""

import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 

import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
  
import tensorflow as tf 
import matplotlib.pyplot as plt

TRAIN = 'C:/Users/tony.george/Documents/Kaggle/flexhose/train'
TEST =  'C:/Users/tony.george/Documents/Kaggle/flexhose/test'

IMG_SIZE = 50
LR = 1e-3

'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'flexihose-{}-{}.model'.format(LR, '6conv-basic') 

def label_img(word_label):
  
 # DIY One hot encoder
    if word_label == 'good': return [1, 0, 0]
    elif word_label == 'frey': return [0, 1, 0]
    elif word_label == 'rust': return [0, 0, 1]


training_data1 = [] 

for dir in os.listdir(TRAIN):
    path = os.path.join(TRAIN, dir)
    for img in tqdm(os.listdir(path)): 
        
        # labeling the images 
        label = label_img(dir) 
      
        path1 = os.path.join(path, img) 
      
        # loading the image from the path and then converting them into 
        # greyscale for easier covnet prob 
        img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE) 
      
        # resizing the image for processing them in the covnet 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
        training_data1.append([np.array(img), np.array(label)]) 

testing_data1 = [] 
for img in tqdm(os.listdir(TEST)): 
    path = os.path.join(TEST, img) 
    img_num = img.split('.')[0] 
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
    testing_data1.append([np.array(img), img_num]) 
    
shuffle(testing_data1) 
np.save('test_data1.npy', testing_data1) 

tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 3, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log') 

train = training_data1[:-3] 
test = training_data1[-3:] 

X1 = np.array([i[0] for i in training_data1]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y1 = [i[1] for i in training_data1] 
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = [i[1] for i in test] 

model.fit({'input': X1}, {'targets': Y1}, n_epoch = 5,  
    validation_set =({'input': test_x}, {'targets': test_y}),  
    snapshot_step = 3, show_metric = True, run_id = MODEL_NAME) 
model.save(MODEL_NAME) 

fig = plt.figure() 
  
for num, data in enumerate(testing_data1): 
    # cat: [1, 0] 
    # dog: [0, 1] 
      
    img_num = data[1] 
    img_data = data[0] 
      
    y = fig.add_subplot(4, 5, num + 1) 
    orig = img_data 
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1) 
  
    # model_out = model.predict([data])[0] 
    model_out = model.predict([data])[0] 
      
    if np.argmax(model_out) == 0: str_label ='good'
    elif np.argmax(model_out) == 1: str_label ='frey'
    else: str_label ='rust'
          
    y.imshow(orig, cmap ='gray') 
    plt.title(str_label) 
    y.axes.get_xaxis().set_visible(False) 
    y.axes.get_yaxis().set_visible(False) 
plt.show() 
