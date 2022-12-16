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
datadir = "C:\Work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV\CV" #path ของ dataset
data = pathlib.Path(datadir)
'''
batch_size = 15
img_height = 150
img_width = 150
'''

batch_size = 32
img_height = 300
img_width = 300

filepath='model2.h5'
filepath_model = 'model2.json'
filepath_weights = 'weights_model2.h5'
#Load
with open('history_model', 'rb') as file:
    his = p.load(file)

predict_model = load_model(filepath)
predict_model.summary()


# Get training and test loss histories
training_loss = his['loss']
test_loss = his['val_loss']
training_accuracy = his['accuracy']
test_accuracy = his['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
fig, (ax1, ax2) = plt.subplots(1, 2)
ax2.plot(epoch_count, training_loss, 'r-')
ax2.plot(epoch_count, test_loss, 'b-')
ax2.legend(['Train Loss', 'Validation Loss'])
ax2.set(xlabel='Epoch',ylabel='Loss Value')
ax2.grid()
ax2.axis([0, 45, 0, 1])
ax1.plot(epoch_count, training_accuracy, 'r-')
ax1.plot(epoch_count, test_accuracy, 'b-')
ax1.legend(['Train Accuracy', 'Validation Accuracy'])
ax1.set(xlabel='Epoch',ylabel='Accuracy Value')
ax1.grid()
ax1.axis([0, 45, 0.75, 1])
plt.show()

'''
h1 = go.Scatter(y=his['loss'],mode='lines', line=dict(width=2,color='blue'),name='loss')
h2 = go.Scatter(y=his['val_loss'],mode='lines', line=dict(width=2,color='red'),name='val_loss')
data = [h1,h2]
layout1 = go.Layout(title='Loss',xaxis=dict(title='epochs'),yaxis=dict(title=' '))
fig1 = go.Figure(data, layout=layout1)
plotly.offline.iplot(fig1, filename='testMNIST')
'''

#predict_model.summary()
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)
    print('Loaded model from disk')


#prediction


test_path = ('0_0_0.59_10.107_1_24_7.054.jpeg')
img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = predict_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print('Battery',score[0],"Pseudocapacitor",score[1])
#display(Image(filename=test_path,width=180, height=180))
#Battery 100%
if score[0]==np.max(score) :    
    PictureExt = 'Battery'
    result = PictureExt
    score_r=score[0]
    score_all=score[0]
#Pseudocapacitor  
elif score[1]==np.max(score) :
    PictureExt = "Pseudocapacitor"
    result = PictureExt
    score_r=score[1]
    score_all=score[1]


imgresize = cv2.imread(test_path)
xHigh,yWidth,z = imgresize.shape
sizeP=800
if yWidth >xHigh :
    imgresize = cv2.resize(imgresize,(sizeP,(yWidth//xHigh*sizeP)))
else :
    imgresize = cv2.resize(imgresize,((xHigh//yWidth)*sizeP,sizeP)) 

cv2.putText(imgresize,f'{result} confident {score_r*100:.2f}%.',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
print('{} confident {:.2f}%.'.format(PictureExt, 100 * np.max(score)))


cv2.imshow("show",imgresize) 
cv2.waitKey(0)
