import tensorflow as tf
from scipy import misc

"""
filename_queue = tf.train.string_input_producer(['8.jpg'])
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_png(value)

#img=my_img.eval()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(my_img.shape)
"""


"""
filenames = ['9.jpg']
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

images = tf.image.decode_jpeg(value, channels=1)

print(images.shape)

"""

import numpy as np
from PIL import Image
from scipy import ndimage
im=Image.open('9.jpg')
img=im.resize((28,28))
img=img.convert('L')
img.save('re9',"JPEG")
img_array=np.array(Image.open('re9')).flatten()
img_array=np.reshape(img_array,(1,784))
print(img_array.shape)
#img_tf = tf.Variable(img)


"""
#works that's exactly what i thought
img = misc.imread('9.jpg')
print (img.shape)    # (32, 32, 3)              

img_tf = tf.Variable(img,dtype=tf.float32)
print (img_tf.get_shape().as_list())  # [32, 32, 3]

print(img_tf)
print(img_tf.shape)
"""

"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())
im = sess.run(img_tf)

import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()

plt.imshow(test_img)
plt.show()
"""



"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

img_list=['4.png','5.jpg','6.jpg','6.png','7.jpg','8.png','9.jpg','1.jpg','two.png','two2.png']


for i in img_list:
    img=misc.imread(i)
    img_tf=tf.Variable(img)
    im = sess.run(img_tf)
    print(im)
    print(im.shape)
"""






