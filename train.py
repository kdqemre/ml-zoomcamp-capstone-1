#!/usr/bin/env python
# coding: utf-8

# # ML-ZOOMCAMP CAPSTONE PROJECT-1

# #### Emre Öztürk

# #### Image Classification Project




import numpy as np
import pandas as pd


import os

import tensorflow as tf
from tensorflow import keras

import cv2
#Load Xception from Keras 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

#Load Xception from Keras 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator





#


path = '/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/dataset/fast-food-classification-v2-small/Train/Baked Potato/'
name = 'Baked Potato-Train (1).jpeg'
fullname = f'{path}/{name}'
img = load_img(fullname)


# `Our raw image dimensions.`




x = np.array(img)
x.shape


# #### We will use pre-trained CNN models with transfer Learning

# `There are many models you can find. I will use Xception because it has small size and high accuracy.`
# * we will train pretrained model for our purpose(for predicting our classes)

# ![Ekran%20Resmi%202022-12-22%2014.39.58.png](attachment:Ekran%20Resmi%202022-12-22%2014.39.58.png)



# We are specifiying we will use pre trained model over 'imagenet' dataset and size of images
model = Xception(weights='imagenet', input_shape=(299, 299, 3))










####################################### ################################################### Model `Version_6` : `299x299` images, `learning rate=0.005`, inner = 100, using drop outs = 0.2 , No Data Augmenation


def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(10)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model




input_size = 299




#we can do data augmentation in ImageDataGenerator
#We apply augmentation training dataset
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    #It chooses randomly between -30, 30 
    #rotation_range=30,
    #It chooses randomly between -10, 10 
    #width_shift_range=10.0,
    #height_shift_range=10.0,
    shear_range=10.0,
    #Between 0.9,1.1
    zoom_range=0.1,
    vertical_flip=True
)

train_ds = train_gen.flow_from_directory(
    '/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/dataset/fast-food-classification-v2-small/Train',
    target_size=(150, 150),
    batch_size=32
)

#We do not apply augmentation on validation dataset. Because it generates random images
#and we want to compare our model with other models
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    '/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/dataset/fast-food-classification-v2-small/Valid',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


# In[65]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v6_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[66]:


learning_rate = 0.0005
size = 100
droprate = 0.2

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])


# ### Conclusion.
# 
# `Not all projects being successfully with less data or bad data.This project shows us we need more data or labeling data for better accuracy. For my limited time i didnt use large data set. You can try. I hope you enjoyed the projetc.`

