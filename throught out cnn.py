import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

image = np.array([[[[1],[29],[57]],
                   [[85],[113],[141]],
                   [[169],[197],[225]]]],dtype=np.float32)

print(image.shape)
plt.imshow(image.reshape(3,3))
plt.show()

weight = tf.constant([[[[0.2]],[[0.2]]],
                     [[[0.2]],[[0.2]]]])

conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding='SAME')
conv2d_img=conv2d.eval()
print("conv2d")
print(conv2d_img)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
print(conv2d_img.shape)
print("maxpool")
pool = tf.nn.max_pool(conv2d_img,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
print(pool.shape)
print(pool.eval())
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3))
    plt.show()


weight = tf.constant([[[[0.2,0.3,0.2]],[[0.2,0.3,0.2]]],
                      [[[0.2,0.3,0.2]],[[0.2,0.3,0.2]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3) )
    plt.show()
