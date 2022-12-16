import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import cv2  
import os
from sklearn.utils import validation
import tensorflow as tf
import PIL
import time
from tensorflow.keras import layers
import pickle as p
import plotly
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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



batch_size = 32
img_height = 224
img_width = 224
num_classes = 2
epochs=60
learningRate= 0.00001


Model ="MobileNetV2_GCD"
Test_label = []
predict_label = [] 


#Prepare data
import pathlib
datadir = "C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\GCD\GCD" #path ของ dataset
data = pathlib.Path(datadir)

#Train #sแบ่งข้อมูล เพื่อ training 80% และ validate 20% 
train = tf.keras.preprocessing.image_dataset_from_directory(data,validation_split=0.2,subset='training',seed=123,image_size=(img_height, img_width),batch_size=batch_size)
val = tf.keras.preprocessing.image_dataset_from_directory(data,validation_split=0.2,subset='validation',seed=123,image_size=(img_height, img_width),batch_size=batch_size)


##Loading base model
base_model = tf.keras.applications.MobileNetV2((img_height, img_width,3),include_top=False,weights="imagenet")
image_batch, label_batch = next(iter(train))
feature_batch = base_model(image_batch)
#base_model.summary()
base_model.trainable = False
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
##Add custom head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer=tf.keras.layers.Dense(units=2,activation="softmax")(global_average_layer)
model = tf.keras.models.Model(inputs=base_model.input,outputs=prediction_layer)
model.summary()



##Train 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),loss="categorical_crossentropy",metrics=["accuracy"])
#his=model.fit(train,epochs=epochs,validation_data=val)


#fine tuning 

print(len(base_model.layers))
base_model.trainable= True
for layer in base_model.layers[:100]:
    layer.trainable = False 


model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningRate/10),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

his=model.fit(train,epochs=epochs,validation_data=val)

training_loss = his.history['loss']
test_loss = his.history['val_loss']
training_accuracy = his.history['accuracy']
test_accuracy = his.history['val_accuracy']



#Save
with open(f'history_model_{Model}', 'wb') as file:
    p.dump(his.history, file)

filepath=f'model1_{Model}.h5'
model.save(filepath)
filepath_model = f'model1_{Model}.json'
filepath_weights = f'weights_model_{Model}.h5'
model_json = model.to_json()
with open(filepath_model, 'w') as json_file:
    json_file.write(model_json)
    model.save_weights(f'weights_model_{Model}.h5')
    print('Saved model to disk')



# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
fig, (ax1, ax2) = plt.subplots(1, 2)
ax2.plot(epoch_count, training_loss, 'r-')
ax2.plot(epoch_count, test_loss, 'b-')
ax2.legend(['Train Loss', 'Validation Loss'])
ax2.set(xlabel='Epoch',ylabel='Loss Value')
ax2.grid()
ax2.axis([0, epochs, 0, 1])

ax1.plot(epoch_count, training_accuracy, 'r-')
ax1.plot(epoch_count, test_accuracy, 'b-')
ax1.legend(['Train Accuracy', 'Validation Accuracy'])
ax1.set(xlabel='Epoch',ylabel='Accuracy Value')
ax1.grid()
ax1.axis([0, epochs, 0.60, 1])

plt.show()

j=1
for images, labels in val.take(-1):
    
    for i in range(len(labels)):
        j+=1
        print(j)
        Test_label.append(labels[i].numpy())
        img_array = keras.preprocessing.image.img_to_array(images[i].numpy())
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions)
        for f in range(len(score)): 
            a=(score[f].numpy())
            if a[0] >= a[1]:
                predict_label.append(0)
            else :
                predict_label.append(1)
        

        
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
