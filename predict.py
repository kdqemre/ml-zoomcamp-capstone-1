#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input



######################################################################################################


# In[19]:



import pickle
from flask import Flask
from flask import request
from flask import jsonify


# In[18]:


#for load the model we use Keras
model = keras.models.load_model('/Users/kadkoy/Desktop/ml-zoomcamp-capstone-project-2022/xception_v4_1_01_0.676.h5')


# In[1]:


app = Flask("food_classiffier")

@app.route("/predict", methods=["POST"])
def predict():
    food_img = request.get_json()
    
    pred = model.predict(food_img)
    
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
    
    
   
    pred_dict = dict(zip(classes, pred[0]))
    
    y_pred = str(pred_dict)
    
        

    result = {
        
        
        "prediction": y_pred
            
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)


# In[ ]:




