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
import matplotlib.font_manager as font_manager
from matplotlib.ticker import ScalarFormatter

#Prepare data

class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

plt.rcParams['font.size'] = '12'
plt.rcParams['font.sans-serif'] = 'Times New Roman'

batch_size = 32
img_height = 224
img_width = 224
X_Axis= 20

K =["VGG16_GCD","VGG16_CV"]
m=1
for model in K :
    m+=1
    Model =model
    
    #Load
    
    if m ==2 :
        with open(f'history_model_{Model}', 'rb') as file:
            his2 = p.load(file)
    elif m==3 :
        with open(f'history_model_{Model}', 'rb') as file:
            his3 = p.load(file)
        


# Get training and test loss histories
training_loss_2 = his2['loss']
test_loss_2 = his2['val_loss']
training_accuracy_2 = his2['accuracy']
test_accuracy_2 = his2['val_accuracy']

training_loss_3 = his3['loss']
test_loss_3 = his3['val_loss']
training_accuracy_3 = his3['accuracy']
test_accuracy_3 = his3['val_accuracy']
# Create count of the number of epochs
epoch_count = range(1, len(training_loss_2) + 1)
epoch_count2 = range(1, len(training_loss_3) + 1)
# Visualize loss history
fig, (ax) = plt.subplots(nrows=3, ncols=2, sharex=True,
                                    figsize=(18,10 ))

font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=14)

ax1 = plt.subplot(2,  2, 1)   
ax1.plot(epoch_count, training_accuracy_2, c='Tab:pink',linewidth=2,linestyle='dashed')
ax1.plot(epoch_count, test_accuracy_2, c='Tab:pink',linewidth=2)
ax1.set_title("GCD Accuracy",fontsize=16,fontname='Arial',weight='bold',pad=10)
ax1.legend(['Train Accuracy', 'Validation Accuracy'],prop=font,loc = 'lower right')
ax1.set_xlabel('Epoch',fontsize=14,weight='bold',fontname='Arial')
ax1.set_ylabel('Accuracy Value',fontsize=14,weight='bold',fontname='Arial') 
ax1.grid(b=True, which='major', linestyle='--')
ax1.set_axisbelow(True)
ax1.axis([0, X_Axis, 0.60, 1])


                                
for t in ax1.get_yticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')
for t in ax1.get_xticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')


ax2 = plt.subplot(2,  2, 3)
ax2.plot(epoch_count, training_loss_2,c='Tab:pink',linewidth=2,linestyle='dashed')
ax2.plot(epoch_count, test_loss_2,  c='Tab:pink',linewidth=2)
ax2.set_title("GCD Loss Curve",fontsize=16,fontname='Arial',weight='bold',pad=10)
ax2.legend(['Train Loss', 'Validation Loss'],prop=font,loc = 'upper right')
ax2.set_xlabel('Epoch',fontsize=14,weight='bold',fontname='Arial')
ax2.set_ylabel('Loss Value',fontsize=14,weight='bold',fontname='Arial') 
ax2.grid(b=True, which='major', linestyle='--')
ax2.set_axisbelow(True)
ax2.axis([0, X_Axis, 0.00, 1.00])

for t in ax2.get_yticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')
for t in ax2.get_xticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')



ax3= plt.subplot(2,  2, 2)
ax3.set_title("CV Accuracy",fontsize=16,fontname='Arial',weight='bold',pad=10)
ax3.plot(epoch_count2, training_accuracy_3,c='Tab:purple',linewidth=2,linestyle='dashed')
ax3.plot(epoch_count2, test_accuracy_3, c='Tab:purple',linewidth=2)
ax3.legend(['Train Accuracy', 'Validation Accuracy'],prop=font,loc = 'lower right')
ax3.set_xlabel('Epoch',fontsize=14,weight='bold',fontname='Arial')
ax3.set_ylabel('Accuracy Value',fontsize=14,weight='bold',fontname='Arial') 
ax3.grid(b=True, which='major', linestyle='--')
ax3.set_axisbelow(True)
ax3.axis([0, X_Axis, 0.60, 1.00])

for t in ax3.get_yticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')
for t in ax3.get_xticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')



ax4 = plt.subplot(2,  2, 4)
ax4.set_title("CV Loss Curve",fontsize=16,fontname='Arial',weight='bold',pad=10)
ax4.plot(epoch_count2, training_loss_3,c='Tab:purple',linewidth=2,linestyle='dashed')
ax4.plot(epoch_count2, test_loss_3, c='Tab:purple',linewidth=2)
ax4.legend(['Train Loss', 'Validation Loss'],prop=font,loc = 'upper right')
ax4.set_xlabel('Epoch',fontsize=14,weight='bold',fontname='Arial')
ax4.set_ylabel('Loss Value',fontsize=14,weight='bold',fontname='Arial') 
ax4.axis([0, X_Axis, 0.00, 1.00])
ax4.grid(b=True, which='major', linestyle='--')
ax4.set_axisbelow(True)

for t in ax4.get_yticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')
for t in ax4.get_xticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')



ax1.text(0.5,0.97,"a",fontsize=20,weight='bold',fontname='Times New Roman')
ax2.text(0.5,0.9,"c",fontsize=20,weight='bold',fontname='Times New Roman')
ax3.text(0.5,0.97,"b",fontsize=20,weight='bold',fontname='Times New Roman')
ax4.text(0.5,0.9,"d",fontsize=20,weight='bold',fontname='Times New Roman')


yScalarFormatter = ScalarFormatterClass(useMathText=True)
yScalarFormatter.set_powerlimits((0,0))
ax4.yaxis.set_major_formatter(yScalarFormatter)
ax2.yaxis.set_major_formatter(yScalarFormatter)
plt.subplots_adjust(hspace = 0.25,wspace = 0.15,top=0.95,bottom=0.06)

plt.show()

'''
h1 = go.Scatter(y=his['loss'],mode='lines', line=dict(width=2,color='blue'),name='loss')
h2 = go.Scatter(y=his['val_loss'],mode='lines', line=dict(width=2,color='red'),name='val_loss')
data = [h1,h2]
layout1 = go.Layout(title='Loss',xaxis=dict(title='epochs'),yaxis=dict(title=' '))
fig1 = go.Figure(data, layout=layout1)
plotly.offline.iplot(fig1, filename='testMNIST')


#predict_model.summary()
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)
    print('Loaded model from disk')


#prediction


test_path = ('0_0_0.99319988489151_10.103_2_5_3600x.png')
img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = predict_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print('Battery',score[0],"Pseudocapacitor",score[1])
#display(Image(filename=test_path,width=180, height=180))
if score[0]==np.max(score) :
    PictureExt = 'Battery'
    result = 'Battery'
    score_r=score[0]

elif score[1]==np.max(score) :
    PictureExt = "Pseudocapacitor"
    result = 'Pseudocapacitor'
    score_r=score[1]


imgresize = cv2.imread(test_path)
xHigh,yWidth,z = imgresize.shape
sizeP=800
imgresize = cv2.resize(imgresize,(sizeP,(yWidth//xHigh*sizeP)))
cv2.putText(imgresize,f'{result} confident {score_r*100:.2f}%.',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
print('{} confident {:.2f}%.'.format(PictureExt, 100 * np.max(score)))


cv2.imshow("show",imgresize) 
cv2.waitKey(0)

'''