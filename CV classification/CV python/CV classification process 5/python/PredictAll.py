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
Model ="ResNet50"

import pathlib
datadir = "C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\CV_Pseudocapacitor" #path ของ dataset
data = pathlib.Path(datadir)
batch_size = 32
img_height = 224
img_width = 224

filepath=f'model1_{Model}.h5'
filepath_model = f'model1_{Model}.json'
filepath_weights = f'weights_model_{Model}.h5'
#Load
with open(f'history_model_{Model}', 'rb') as file:
    his = p.load(file)


predict_model = load_model(filepath)
#predict_model.summary()
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)
    #print('Loaded model from disk')


j=0

path = "C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\All_Pseudocapacitor"
files = os.listdir(path)
for f in files:
    #print(f)
    j=j+1
    print(j," in ",len(files))
   
    #prediction
    test_path = f"{path}\{f}"
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    #print('Pseudocapacitor',score[0],"Pseudocapacitor50",score[1])
    #display(Image(filename=test_path,width=180, height=180))
    imgresize = cv2.imread(test_path)   # load CV2 img
    
    #Pseudocapacitor 100%
   
    if score[0]==np.max(score) :    
        PictureExt = score[0]/75*100
        PictureN= "0"

        if PictureExt>=0.95:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/100/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
            #os.remove(f"{path}\{f}")  
            #print(f"confident {score[0]} of ",{PictureExt})
        if PictureExt>=0.85 and PictureExt<0.95:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/90/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
            #os.remove(f"{path}\{f}")  
            #print(f"confident {score[0]} of ",{PictureExt})
        if PictureExt>=0.75 and PictureExt<0.85:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/80/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
            #os.remove(f"{path}\{f}")  
            #print(f"confident {score[0]} of ",{PictureExt})
        if PictureExt>=0.65 and PictureExt<0.75:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/70/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
            #os.remove(f"{path}\{f}")  
            #print(f"confident {score[0]} of ",{PictureExt})
        if PictureExt>=0.55 and PictureExt<0.65:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/60/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
            #os.remove(f"{path}\{f}")  
            #print(f"confident {score[0]} of ",{PictureExt})
        if PictureExt>=0.45 and PictureExt<0.55:
            cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/50/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
            #os.remove(f"{path}\{f}")  
            #print(f"confident {score[0]} of ",{PictureExt})

    #Pseudocapacitor 50%
    elif score[1]==np.max(score) :    
            PictureExt = 0.5*(1-(score[1]/75*100)+1)
            
            PictureN= "1"
            if PictureExt>=0.95: 
                cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/100/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
                #os.remove(f"{path}\{f}")  
                #print(f"confident {score[1]} of ",{PictureExt})
            if PictureExt>=0.85 and PictureExt<0.95:
                cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/90/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
                #os.remove(f"{path}\{f}")  
                #print(f"confident {score[1]} of ",{PictureExt})
            if PictureExt>=0.75 and PictureExt<0.85:
                cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/80/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
                #os.remove(f"{path}\{f}")  
                #print(f"confident {score[1]} of ",{PictureExt})
            if PictureExt>=0.65 and PictureExt<0.75:
                cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/70/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
                #os.remove(f"{path}\{f}")  
                #print(f"confident {score[1]} of ",{PictureExt})
            if PictureExt>=0.55 and PictureExt<0.65:
                cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/60/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
                #os.remove(f"{path}\{f}")  
                #print(f"confident {score[1]} of ",{PictureExt})
            if PictureExt>=0.45 and PictureExt<0.55:
                cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/50/{PictureN}_{PictureExt*100:.2f}_{f}",imgresize)
                #os.remove(f"{path}\{f}")  
                #print(f"confident {score[1]} of ",{PictureExt})

    '''
    xHigh,yWidth,z = imgresize.shape
    sizeP=800
    if yWidth >xHigh :
        imgresize = cv2.resize(imgresize,(sizeP,(yWidth//xHigh*sizeP)))
    else :
        imgresize = cv2.resize(imgresize,((xHigh//yWidth)*sizeP,sizeP)) 

    '''
    #cv2.putText(imgresize,f'{result} confident {score_r*100:.2f}%.',(20,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
    #print('{} confident {:.2f}%.'.format(PictureExt, 100 * np.max(score)))
    #cv2.imshow("show",imgresize) 
    #cv2.imwrite(f"D:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV/OutputCVall\{result}_{score_r*100:.2f}_{score_all*100:.2f}_{PictureExt}_{f}",imgresize)


            
                    
    

