
from operator import mod
import os
from pdfrw import PdfReader
import shutil
import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os
import tensorflow as tf
from tensorflow.keras import layers
import pickle as p
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras.models import Sequential


def loadModel(predict_model_1):
    test_path = (image_path)
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model_1.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score

def model1():
    global img_height,img_width
    img_height = 224
    img_width = 224

def model2():
    global img_height,img_width
    img_height = 224
    img_width = 224
    

#Load_model1
Model='ResNet50_picture'  
filepath=f'model1_{Model}.h5'
filepath_model = f'model1_{Model}.json'
filepath_weights = f'weights_model_{Model}.h5'
#Load
with open(f'history_model_{Model}', 'rb') as file:
    his = p.load(file) 
predict_model1 = load_model(filepath)
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model_1 = model_from_json(loaded_model_json)
    predict_model_1.load_weights(filepath_weights)

#Load_model2
Model ="ResNet50_GCD"
filepath=f'model1_{Model}.h5'
filepath_model = f'model1_{Model}.json'
filepath_weights = f'weights_model_{Model}.h5'
#Load
with open(f'history_model_{Model}', 'rb') as file:
    his = p.load(file)
predict_model2 = load_model(filepath)
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model_2 = model_from_json(loaded_model_json)
    predict_model_2.load_weights(filepath_weights)
#Load_model3
Model ="ResNet50_CV"
filepath=f'model1_{Model}.h5'
filepath_model = f'model1_{Model}.json'
filepath_weights = f'weights_model_{Model}.h5'
#Load
with open(f'history_model_{Model}', 'rb') as file:
    his = p.load(file)
predict_model_3 = load_model(filepath)
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model_3 = model_from_json(loaded_model_json)
    predict_model_3.load_weights(filepath_weights)

path = r'montree_bibliographie_21 fevrier 2021'
files = os.listdir(path)


for j in range(0,len(files)):
    z= (files[j])
    fullName = os.path.join(path, z)
    
    newfolder = os.path.join(path, z[:-4])
    newfolder_Picall=os.path.join(newfolder, "picture_all")
    newfolder_PicExtract=os.path.join(newfolder, "picture_Extract")  

    try :
        os.mkdir(newfolder)
        os.mkdir(newfolder_Picall)
        os.mkdir(newfolder_PicExtract)
        
    except:
        pass
    print(j ," in ", len(files) )



    try:
        pdf_file = fitz.open(fullName)
    
        # iterate over PDF pages
        for page_index in range(len(pdf_file)):
            # get the page itself
            page = pdf_file[page_index]
            image_list = page.get_images() 
            for image_index, img in enumerate(page.get_images(), start=1):
                # get the XREF of the image
                xref = img[0]
                # extract the image bytes
                base_image = pdf_file.extract_image(xref)
                try:
                    image_bytes = base_image["image"]
                except:
                    continue
                # get the image extension
                image_ext = base_image["ext"]
                # load it to PIL
                image = Image.open(io.BytesIO(image_bytes))
                # save it to local disk
                local  = f"{newfolder_Picall}/test.{image_ext}"
                #local  = f"{newfolder_Picall}/{page_index+1}_{image_index}.{image_ext}"

                try:
                    image.save(open(local, "wb"))
                    imgresize = cv2.imread(local)
                    imgresize2 = cv2.imread(local)
                except:
                    continue
            
                # Grayscale
                gray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray,(15,15),0)
                thresh=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
                kernel=np.ones((3,3),np.uint8)
                closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=6)
                result_img=closing.copy()
                contours,hierachy=cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                i=0
                for cnt in contours:
                    i=i+1
                    area = cv2.contourArea(cnt)
                
                    if area > 10000: 
                        (x,y,w,h)= cv2.boundingRect(cnt)
                        cv2.rectangle(imgresize,(x,y),(x+w,y+h),(0,0,255),2)
                        result = imgresize2[y:y+h,x:x+w]
                        local_save = f"{newfolder_PicExtract}/Test.{image_ext}"
                        #local_save = f"{newfolder_PicExtract}/{page_index+1}_{image_index}_{i}.{image_ext}"
                        cv2.imwrite(local_save,result)
                    
                        #prediction
                        #import requests
                        #from io import BytesIO
                        model1()
                        image_path = local_save
                        score= loadModel(predict_model_1)
                        #print('CV',score[0],"GCD",score[1],"Other",score[2])
                        if score[0]==np.max(score) :
                            PictureExt = 'CV'
                            model2()
                            score= loadModel(predict_model_3)
                            if score[0]==np.max(score) :
                                PictureInt = "Bat"
                                label = 0
                                cv2.imwrite(f"{newfolder}/{label}_{PictureInt}_CV_0_0_{score[0]/75*100:.2f}_{page_index+1}_{image_index}_{(i)}_{z[:-4]}.{image_ext}",result)
                            elif score[1]==np.max(score) :
                                PictureInt = "Pse"
                                label = 1
                                cv2.imwrite(f"{newfolder}/{label}_{PictureInt}_CV_0_0_{score[1]/75*100:.2f}_{page_index+1}_{image_index}_{(i)}_{z[:-4]}.{image_ext}",result)
                        elif score[1]==np.max(score) :
                            PictureExt = "GCD"
                            model2()
                            score= loadModel(predict_model_2)
                            if score[0]==np.max(score) :
                                PictureInt = "Bat"
                                label = 0
                                cv2.imwrite(f"{newfolder}/{label}_{PictureInt}_GCD_0_0_{score[0]/75*100:.2f}_{page_index+1}_{image_index}_{(i)}_{z[:-4]}.{image_ext}",result)
                            elif score[1]==np.max(score) :
                                PictureInt = "Pse"
                                label = 1
                                cv2.imwrite(f"{newfolder}/{label}_{PictureInt}_GCD_0_0_{score[1]/75*100:.2f}_{page_index+1}_{image_index}_{(i)}_{z[:-4]}.{image_ext}",result)
                        elif score[2]==np.max(score) :
                            pass
    except:
        pass                     

    
    # Extract pdf title from pdf file
    try :
        newName = PdfReader(fullName).Info.Title

        
        newName = newName.strip('()')+'.pdf'
        #print(newName)
        #newName = newName.replace('a', '')
        newFullName = os.path.join(newfolder, z)
        shutil.copyfile(fullName, newFullName)

    except  :
        continue

    


    # Remove surrounding brackets that some pdf titles have
    
    