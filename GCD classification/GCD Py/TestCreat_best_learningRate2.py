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
import warnings
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
print( 'Tensorflow Version:', tf.__version__)
print('GPU Available::', tf.config.list_physical_devices('GPU'))

#Prepare data
import pathlib
datadir = "C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\GCD\GCD" #path ของ dataset
data = pathlib.Path(datadir)
image_count = len(list(data.glob('*/*.jpeg'))) #ขนาดของ dataset
print("จำนวนรูปภาพที่มี .jpg : ",image_count)

batch_size = 32
img_height = 300
img_width = 300
num_classes = 2
epochs=100
learningRate= 0.00001

#Train #sแบ่งข้อมูล เพื่อ training 80% และ validate 20% validation_split=0.3
train = tf.keras.preprocessing.image_dataset_from_directory(data,validation_split=0.2,subset='training',seed=123,image_size=(img_height, img_width),batch_size=batch_size)
val = tf.keras.preprocessing.image_dataset_from_directory(data,validation_split=0.2,subset='validation',seed=123,image_size=(img_height, img_width),batch_size=batch_size)
#Data set
class_names = train.class_names
print(class_names)

#Normalization
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

tf.random.set_seed(42)
initial_model = Sequential([
layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
layers.Conv2D(16, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(num_classes)
])

initial_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history_optimized = initial_model.fit(train,validation_data=val,epochs=epochs)

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False 

plt.plot(
    np.arange(1, 101), 
    history_optimized.history['val_loss'], 
    label='Loss', lw=3
)
plt.plot(
    np.arange(1, 101), 
    history_optimized.history['loss'], 
    label='Accuracy', lw=3
)
plt.title('Accuracy vs. Loss per epoch', size=20)
plt.xlabel('Epoch', size=14)
plt.legend()
plt.show()
'''

learning_rates = 1e-3 * (10 ** (np.arange(100) / 30))
plt.semilogx(
    learning_rates, 
    initial_history.history['loss'], 
    lw=3, color='#000'
)
plt.title('Learning rate vs. loss', size=20)
plt.xlabel('Learning rate', size=14)
plt.ylabel('Loss', size=14)
plt.show()
'''