import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np
import os
from datetime import datetime 

st.header('Image Classification Model')

# Clear cache and reload model
st.cache_resource.clear()

@st.cache_resource
def load_classification_model():
    model_path = r'C:\Users\ABHINAV KUMAR\Desktop\Image Classification\Image_classify.keras'
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    st.write(f"Loading model last modified: {mod_time}")
    return load_model(model_path)

model = load_classification_model()

# Add a button to clear cache and reload model
if st.button('Reload Model'):
    st.cache_resource.clear()
    model = load_classification_model()
    st.success('Model reloaded!')

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180
image =st.text_input('Enter Image name','Apple.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.img_to_array(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))