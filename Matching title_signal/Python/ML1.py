
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

#Prepare data
import pathlib
datadir = "PictureExt" #path ของ dataset
data = pathlib.Path(datadir)
image_count = len(list(data.glob('*/*.jpg'))) #ขนาดของ dataset
print("จำนวนรูปภาพที่มี .jpg : ",image_count)

batch_size = 32
img_height = 150
img_width = 150
num_classes = 3
epochs=100
learningRate=0.00001

#Train
train = tf.keras.preprocessing.image_dataset_from_directory(
data,
validation_split=0.2, #sแบ่งข้อมูล เพื่อ training 80% และ validate 20%
subset='training',
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
val = tf.keras.preprocessing.image_dataset_from_directory(
data,
validation_split=0.2,
subset='validation',
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
#Data set
class_names = train.class_names
print(class_names)

#Display
#import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()

for image_batch, labels_batch in train:
    print(image_batch.shape)
    print(labels_batch.shape)
    break  

#Normalization
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

#model 

model = Sequential([
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

opt = tf.keras.optimizers.Adam(learning_rate=learningRate)
var1 = tf.Variable(10.0)
loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
step_count = opt.minimize(loss, [var1]).numpy()
# The first step is `-learning_rate*sign(grad)`
var1.numpy()
    
model.compile(optimizer=opt,
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
model.summary()

his = model.fit(train,validation_data=val,epochs=epochs)


#Save
with open('history_model', 'wb') as file:
    p.dump(his.history, file)

filepath='model1.h5'
model.save(filepath)
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'
model_json = model.to_json()
with open(filepath_model, 'w') as json_file:
    json_file.write(model_json)
    model.save_weights('weights_model.h5')
    print('Saved model to disk')


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
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

