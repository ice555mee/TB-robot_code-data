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


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#print( 'Tensorflow Version:', tf.__version__)
#print('GPU Available::', tf.config.list_physical_devices('GPU'))

import pathlib
datadir = "C:\Work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV\CV" #path ของ dataset
data = pathlib.Path(datadir)
batch_size = 32
img_height = 300
img_width = 300
filepath='model2.h5'  
filepath_model = 'model2.json'
filepath_weights = 'weights_mode2.h5'
#Load
with open('history_model', 'rb') as file:
    his = p.load(file)
predict_model = load_model(filepath)

j=0
path = "C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV/UnknownCVall"
files = os.listdir(path)

for f in files:
    #print(f)
    j=j+1
    print(j," in ",len(files))
   
    #prediction
    test_path = f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV/UnknownCVall\{f}"
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    #print('Battery',score[0],"Battery50",score[1],"Battery75",score[2],"Pseudocapacitor",score[3],"Pseudocapacitor75",score[4])
    #display(Image(filename=test_path,width=180, height=180))
    imgresize = cv2.imread(test_path)   # load CV2 img
    #Battery 100%
    Score_the=0.80
    if score[0]==np.max(score) :    
        PictureExt = 'Battery'
        result = 'Battery'
        score_r=score[0]
        score_all=score[0]
        if score[0]>Score_the:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV\CV/{PictureExt}/{f}",imgresize)
            os.remove(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV/UnknownCVall/{f}")  
            print(f"confident {score[0]} of ",{PictureExt})    
    #Pseudocapacitor 100%    
    elif score[1]==np.max(score) :
        PictureExt = "Pseudocapacitor"
        result = "Pseudocapacitor"
        score_r=score[1]
        score_all=score[1]
        if score[1]>Score_the:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV\CV/{PictureExt}/{f}",imgresize)
            os.remove(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy_Model2\CV/UnknownCVall/{f}")
            print(f"confident {score[1]} of ",{PictureExt})     
   
    
    xHigh,yWidth,z = imgresize.shape
    sizeP=800
    if yWidth >xHigh :
        imgresize = cv2.resize(imgresize,(sizeP,(yWidth//xHigh*sizeP)))
    else :
        imgresize = cv2.resize(imgresize,((xHigh//yWidth)*sizeP,sizeP)) 
    #cv2.putText(imgresize,f'{result} confident {score_r*100:.2f}%.',(20,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
    #print('{} confident {:.2f}%.'.format(PictureExt, 100 * np.max(score)))
    #cv2.imshow("show",imgresize) 
    #cv2.imwrite(f"D:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV/OutputCVall\{result}_{score_r*100:.2f}_{score_all*100:.2f}_{PictureExt}_{f}",imgresize)


            
                    
    

