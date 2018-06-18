import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#tf.set_random_seed()
#np.random.seed()
batch_size=128
training_epoch=2

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
X_train = mnist.train.images #(55000, 784) 28*28*1
Y_train = mnist.train.labels #(55000, 10)
X_test = mnist.test.images #(10000, 784)
Y_test = mnist.test.labels #(10000, 10)


X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])
X_reshape=tf.reshape(X,[-1,28,28,1])

W1=tf.Variable(tf.random_normal([14,14,1,32],stddev=0.01))
L1=tf.nn.conv2d(X_reshape,W1,strides=[1,1,1,1],padding='SAME')
L1=tf.nn.relu(L1)
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W2=tf.Variable(tf.random_normal([7,7,32,64],stddev=0.01))
L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W3=tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
L3=tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
L3=tf.nn.relu(L3)
L3=tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L3_flat = tf.reshape(L3,[-1,4*4*128])

W4=tf.get_variable("W4",shape=[4*4*128,10],initializer=tf.contrib.layers.xavier_initializer())
b=tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_flat,W4)+b

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=Y))

optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epoch):
    ac=0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        bx,by = mnist.train.next_batch(batch_size)
        feed_dict = {X:bx,Y:by}
        c,_ =sess.run([cost,optimizer],feed_dict=feed_dict)
        ac += c/total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(ac))

correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


print('Accuracy:', sess.run(acc, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))


for i in range(10):
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))


img_list=['4.png','5.jpg','6.jpg','6.png','7.jpg','8.png','9.jpg','1.jpg','two.png','two2.png']

#need to feed (1, 784)


#1.image reshape.
#2.put image to tensor.
#3.feed it.

import numpy as np
from PIL import Image
from scipy import ndimage
im=Image.open('5.jpg') #6.png
img=im.resize((28,28))
img=img.convert('L')
img.save('re9',"JPEG")
img_array=np.array(Image.open('re9')).flatten()
plt.imshow(img_array.reshape(28,28))
plt.show()
img_array=np.reshape(img_array,(1,784))

print("Label: ", sess.run(tf.argmax(img_array, 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: img_array}))

    
