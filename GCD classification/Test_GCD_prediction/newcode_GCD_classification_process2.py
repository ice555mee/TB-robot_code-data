import numpy as np
import cv2  
import os
import tensorflow as tf
from tensorflow.keras import layers
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import plotly.graph_objs as go
from tensorflow import keras
from tensorflow.keras.models import Sequential


def loadModel(predict_model_1):
    test_path = image_path
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model_1.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score


img_height = 224
img_width = 224
#Load_model2
Model ="ResNet50_GCD"
filepath_model = f'model1_{Model}.json'
filepath_weights = f'weights_model_{Model}.h5'
#Load

with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model_2 = model_from_json(loaded_model_json)
    predict_model_2.load_weights(filepath_weights)  

image_path = r"C:\Users\ice_d\Desktop\Test_GCD_prediction\unknownGCD\unknownGCD1.jpeg"
readimg = cv2.imread(image_path)
# new Image directory
newdirec= r"C:\Users\ice_d\Desktop\Test_GCD_prediction\resultGCD"
newloca=os.chdir(newdirec)  
    
    #prediction
   
score= loadModel(predict_model_2)

# New filename
newfilename = 'predicted'

if score[0]==np.max(score) :
    PictureInt = "Battery"
    label = 0
    # Saving the image
    cv2.imwrite(f'{newfilename}_{label}_{PictureInt}_0_{2*score[0]-0.5}.jpg', readimg)
    
elif score[1]==np.max(score) :
    PictureInt = "Pseudocapacitor"
    label = 1
    cv2.imwrite(f'{newfilename}_{label}_{PictureInt}_0_{2*score[1]-0.5}.jpg', readimg)
    
print('complete run')


  
    





