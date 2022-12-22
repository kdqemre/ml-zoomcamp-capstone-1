#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np


# In[3]:


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[4]:


test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    '/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/dataset/fast-food-classification-v2-small/Test',
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)


# In[5]:


#for load the model we use Keras
model = keras.models.load_model('/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/xception_v4_1_01_0.676.h5')


# In[6]:


model.evaluate(test_ds)


# In[7]:


#Path for single image for test
path = '/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/dataset/fast-food-classification-v2-small/Test/Hot Dog/Hot Dog-Test (16).jpeg'


# In[8]:


#Preprocessing image for test
img = load_img(path, target_size=(299, 299))


# In[10]:


x = np.array(img)
X = np.array([x])
X.shape


# In[11]:


X = preprocess_input(X)


# In[12]:


pred = model.predict(X)


# In[ ]:


{'Baked Potato': 0,
 'Burger': 1,
 'Crispy Chicken': 2,
 'Donut': 3,
 'Fries': 4,
 'Hot Dog': 5,
 'Pizza': 6,
 'Sandwich': 7,
 'Taco': 8,
 'Taquito': 9}


# In[13]:


classes = [
    'Baked Potato',
 'Burger',
 'Crispy Chicken',
 'Donut',
 'Fries',
 'Hot Dog',
 'Pizza',
 'Sandwich',
 'Taco',
 'Taquito'
]


# In[14]:


dict(zip(classes, pred[0]))


# In[17]:


######################################################################################################


# In[19]:



import pickle
from flask import Flask
from flask import request
from flask import jsonify


# In[18]:


#for load the model we use Keras
model = keras.models.load_model('/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/xception_v4_1_01_0.676.h5')


# In[ ]:


app = Flask("food_classifier")

@app.route("/predict", methods=["POST"])
def predict():
    food = request.get_json()
    
    X_food = 
   
    X_food = dv.transform([food])
    y_pred = model.predict(X_food)

    y_prob = model.evaluate(X_food)

    
        

    result = {
        
        
        "prediction": float(y_prob)
            
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

