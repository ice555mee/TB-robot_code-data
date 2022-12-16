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
print( 'Tensorflow Version:', tf.__version__)
print('GPU Available::', tf.config.list_physical_devices('GPU'))
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

path = "PaperwithoutCopy"
files = os.listdir(path)
#N 7866
j,N=1,len(files)
print(len(files))

for j in range(j,N):
    z= (files[j])
    print(j , "in " , len(files))
    local_O  = f"{path}/{z}"
    try:
        pdf_file = fitz.open(local_O)
    except:
        print( " cannot open ",f"{z}")
        continue # skip this  and proceed to the next 
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.getImageList() 
        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extractImage(xref)
            try:
                image_bytes = base_image["image"]
            except:
                continue
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to local disk
            local  = f"PictureAll/Test.{image_ext}"
            try:
                image.save(open(local, "wb"))
                imgresize = cv2.imread(local)
                imgresize2 = cv2.imread(local)
            except :
                continue # skip this  and proceed to the next 
            
            # Grayscale
            gray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray,(15,15),0)
            thresh=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
            kernel=np.ones((3,3),np.uint8)
            closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=9)
            result_img=closing.copy()
            contours,hierachy=cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            i=0
            for cnt in contours:
                i=i+1
                area = cv2.contourArea(cnt)
            
                if area > 20000: 
                
                    (x,y,w,h)= cv2.boundingRect(cnt)
                    cv2.rectangle(imgresize,(x,y),(x+w,y+h),(0,0,255),2)
                    result = imgresize2[y:y+h,x:x+w]
                    try:
                        cv2.imwrite(f"PictureExt/test.{image_ext}",result)
                    except :
                        continue

                    import pathlib
                    datadir = "PictureExt" #path ของ dataset
                    data = pathlib.Path(datadir)
                
                    #prediction
                    #import requests
                    #from io import BytesIO
                    test_path = (f'PictureExt/test.{image_ext}')
                    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) # Create a batch
                    predictions = predict_model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    #print('CV',score[0],"GCD",score[1],"Other",score[2])
                    if score[0]==np.max(score) :
                        PictureExt = 'CV'
                        cv2.imwrite(f"PictureExt/CV/0_0_{score[0]:.2f}_{z[:5]}{page_index+1}_{image_index}_{(i)}_{z[-9:-4]}.{image_ext}",result)
                    elif score[1]==np.max(score) :
                        PictureExt = "GCD"  
                        cv2.imwrite(f"PictureExt/GCD/0_0_{score[1]:.2f}_{z[:5]}{page_index+1}_{image_index}_{(i)}_{z[-9:-4]}.{image_ext}",result)
                    elif score[2]==np.max(score) :
                        PictureExt = "Other" 
                        if score[2] <0.4:
                            cv2.imwrite(f"PictureExt/0_0_{score[2]:.2f}_{z[:5]}{page_index+1}_{image_index}_{(i)}_{z[-9:-4]}.{image_ext}",result)
                        
                    #print('{} confident {:.2f}%.'.format(PictureExt, 100 * np.max(score)))
                    

print(j+1)