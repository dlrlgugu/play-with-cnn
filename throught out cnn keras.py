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

def visualize_step(cat_batch):
    cat = np.squeeze(cat_batch,axis=0)
    print (cat.shape)
    plt.imshow(cat)
    plt.show()

def more_clear(model,cat):
    cat_batch=np.expand_dims(cat,axis=0)
    conv_cat2=model.predict(cat_batch)

    conv_cat2=np.squeeze(conv_cat2,axis=0)
    print(conv_cat2.shape)
    #conv_cat2=conv_cat2.reshape(conv_cat2.shape[:2])

    print(conv_cat2.shape)
    plt.imshow(conv_cat2)
    plt.show()

cat = cv2.imread('cat_1.jpg')
plt.imshow(cat)
#plt.show()

model=Sequential()
model.add(Conv2D(3,3,strides=(1,1),padding='same',input_shape=cat.shape))

#cat_batch=np.expand_dims(cat,axis=0)#(350, 467, 3) -> (1, 350, 467, 3)
#conv_cat = model.predict(cat_batch)
#visualize_step(conv_cat)

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

cat_batch=np.expand_dims(cat,axis=0)
conv_cat2=model.predict(cat_batch)
conv_cat2=np.squeeze(conv_cat2,axis=0)
print(conv_cat2.shape)
plt.imshow(conv_cat2)
plt.show()
#more_clear(model,cat)

#more_clear(model,cat)
#print(conv_cat.shape)#(1, 348, 465, 3)


model.add(Conv2D(5,3,padding='valid',input_shape=conv_cat2.shape))
#more_clear(model,cat)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))


cat_batch=np.expand_dims(conv_cat2,axis=0)
conv_cat3=model.predict(cat_batch)
conv_cat3=np.squeeze(conv_cat3,axis=0)
print(conv_cat3.shape)
plt.imshow(conv_cat3)
plt.show()


model.add(Conv2D(15,3,padding='valid',input_shape=conv_cat2.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

cat_batch=np.expand_dims(conv_cat3,axis=0)
conv_cat4=model.predict(cat_batch)
conv_cat4=np.squeeze(conv_cat4,axis=0)
print(conv_cat4.shape)
plt.imshow(conv_cat4)
plt.show()


model.add(Conv2D(55,5,padding='valid',input_shape=conv_cat2.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))


cat_batch=np.expand_dims(conv_cat4,axis=0)
conv_cat5=model.predict(cat_batch)
conv_cat5=np.squeeze(conv_cat5,axis=0)
print(conv_cat5.shape)
plt.imshow(conv_cat5)
plt.show()


model.add(Conv2D(155,10,padding='valid',input_shape=conv_cat2.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

cat_batch=np.expand_dims(conv_cat5,axis=0)
conv_cat6=model.predict(cat_batch)
conv_cat6=np.squeeze(conv_cat6,axis=0)
print(conv_cat6.shape)
plt.imshow(conv_cat6)
plt.show()







