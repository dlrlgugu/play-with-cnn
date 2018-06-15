import tensorflow as tf

file_test = tf.train.string_input_producer(['9.jpg'])

reader = tf.WholeFileReader()
key, value = reader.read(file_test)

img = tf.image.decode_jpeg(value)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

Image.show(img)
