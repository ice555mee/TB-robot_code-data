import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os
import tensorflow as tf
import PIL
import time
from tensorflow.keras import layers
import pickle as p
import plotly
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import plotly.graph_objs as go
from tensorflow import keras
from tensorflow.keras.models import Sequential

#Prepare data
import pathlib
datadir = "PictureExt" #path ของ dataset
data = pathlib.Path(datadir)

batch_size = 15
img_height = 150
img_width = 150


filepath='model1.h5'
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'
#Load
with open('history_model', 'rb') as file:
    his = p.load(file)

predict_model = load_model(filepath)
#predict_model.summary()


# Get training and test loss histories
training_loss = his['loss']
test_loss = his['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history

plt.plot(epoch_count, training_loss, 'r-')
plt.plot(epoch_count, test_loss, 'b-')
plt.axis([0, 100, 0, 1])
plt.legend(['Training Loss', 'validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()
'''
h1 = go.Scatter(y=his['loss'],mode='lines', line=dict(width=2,color='blue'),name='loss')
h2 = go.Scatter(y=his['val_loss'],mode='lines', line=dict(width=2,color='red'),name='val_loss')
data = [h1,h2]
layout1 = go.Layout(title='Loss',xaxis=dict(title='epochs'),yaxis=dict(title=' '))
fig1 = go.Figure(data, layout=layout1)
plotly.offline.iplot(fig1, filename='testMNIST')

predict_model.summary()
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)
    print('Loaded model from disk')
'''


#prediction
import requests
from IPython.display import Image
from io import BytesIO
test_path = ('PictureExt/test.jpeg')
img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = predict_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print('CV',score[0],"GCD",score[1],"Other",score[2])

#display(Image(filename=test_path,width=180, height=180))

if score[0]==np.max(score) :
    PictureExt = 'CV'
elif score[1]==np.max(score) :
    PictureExt = "GCD"  
elif score[2]==np.max(score) :
    PictureExt = "Other"   
print('{} confident {:.2f}%.'.format(PictureExt, 100 * np.max(score)))
imgresize = cv2.imread(test_path)
cv2.imshow("show",imgresize) 
cv2.waitKey(0)