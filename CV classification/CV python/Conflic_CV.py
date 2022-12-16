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
import itertools
from typing import DefaultDict
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score

path = "Battery"
path2 = "Pseudocapacitor"
files = os.listdir(path)
files2 = os.listdir(path2)
Test_value  = []
Test_label = []
predict_label = [] 

#Prepare data
import pathlib
datadir = "C:\Work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV\CV" #path ของ dataset
data = pathlib.Path(datadir)

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
#predict_model.summary()


#predict_model.summary()
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)
    #print('Loaded model from disk')

i=0
for f in files :
    i+=1
    #prediction
    test_path = (f'Battery/{f}')
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    #Battery 100%
    if score[0]==np.max(score) :    
        PictureExt = 'Battery'
        result = PictureExt
        score_r=score[0]
        predict_label.append(0)
        
    #Pseudocapacitor  
    elif score[1]==np.max(score) :
        PictureExt = "Pseudocapacitor"
        result = PictureExt
        score_r=score[1]
        predict_label.append(1)

    Test_value.append(f)
    Test_label.append(0)
    print(i)

for f in files2 :
    i+=1
    #prediction
    test_path = (f'Pseudocapacitor/{f}')
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

       #Battery 100%
    if score[0]==np.max(score) :    
        PictureExt = 'Battery'
        result = PictureExt
        score_r=score[0]
        predict_label.append(0)        
    #Pseudocapacitor  
    elif score[1]==np.max(score) :
        PictureExt = "Pseudocapacitor"
        result = PictureExt
        score_r=score[1]
        predict_label.append(1)

    Test_value.append(f)
    Test_label.append(1)
    print(i)
#print(predict_label)

cm=confusion_matrix(Test_label, predict_label)
print(cm)

group_names = ["TP","FP","FN","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
label_font = {'size':'18'}
sb.heatmap(cm, annot=labels,fmt='', cmap='Blues')
classes=["Battery","Pseudocapacitor"]
plt.imshow(cm,interpolation='nearest',cmap=plt.cm.GnBu)
plt.title("Confusion Matrix") 
trick_marks=np.arange(len(classes))
plt.xticks(trick_marks+0.5,classes)
plt.yticks(trick_marks+0.5,classes)
    
plt.tight_layout()
plt.ylabel('Prediction')
plt.xlabel('Label')
#plt.savefig(f"D:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\Picture/ConfusionMatrix_CV_GCD_%.jpeg")
plt.show()






#classification_report 
print(classification_report(Test_label, predict_label,digits=4,target_names=["Battery","Pseudocapacitor"]))
print("accuracy ", accuracy_score(Test_label, predict_label)*100)
print("Specificity ", cm[1,1]/(cm[0,1]+cm[1,1]))

        