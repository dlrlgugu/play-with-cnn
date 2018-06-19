from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def visualize_step(model , obj):
    exp = np.expand_dims(obj,axis=0)
    conv=model.predict(exp)
    conv=np.squeeze(conv,axis=0)
    print(conv.shape)
    plt.imshow(conv)
    plt.show()
    return conv


cat = cv2.imread('cat_1.jpg')
model=Sequential()
model.add(Conv2D(3,3,strides=(1,1),padding='same',input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

conv = visualize_step(model,cat)

model.add(Conv2D(5,5,padding='valid',input_shape=conv.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

conv = visualize_step(model,conv)

model.add(Conv2D(15,7,padding='valid',input_shape=conv.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

conv = visualize_step(model,conv)

model.add(Conv2D(30,9,padding='valid',input_shape=conv.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

conv = visualize_step(model,conv)

model.add(Conv2D(60,12,padding='valid',input_shape=conv.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

conv = visualize_step(model,conv)

model.add(Conv2D(90,24,padding='valid',input_shape=conv.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

conv = visualize_step(model,conv)


