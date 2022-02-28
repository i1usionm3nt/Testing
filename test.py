from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import data_flow_ops
import tensorflow as tf
from scipy import misc
import numpy as np
import importlib
import facenet
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


filepath = './data/cut/'

# data process

# file_contents = tf.read_file(filename + picname)
# image = tf.image.decode_image(file_contents, channels=3)
# image = tf.random_crop(image, [image_size, image_size, 3])
# image = tf.cast(image, tf.float32)
# image.set_shape((image_size, image_size, 3))
# image = sess.run(image)
random_crop = False
random_flip = False
image_size = 640
batch_size = 1
weight_decay = 1e-05
keep_probability = 0.8
embedding_size = 128
model_def = 'models.inception_resnet_v1_attention'
network = importlib.import_module(model_def)
# model load
path = './models/facenet/'
series = '20210812-201246'
# x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
# _ = tf.train.import_meta_graph(path + '20210818-164038/model-20210818-164038.meta')
# summary_write = tf.summary.FileWriter("./graph" , graph)

# model = network(inputs=x)

class_list = os.listdir(filepath)
with tf.Session() as sess:

    saver = tf.train.import_meta_graph(path + series + '/model-' + series + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(path + series + "/"))
    graph = tf.get_default_graph()

    for i in range(len(class_list)):
        image_names = os.listdir(filepath + class_list[i] + "/")

        for no, j in enumerate(image_names):

            images = np.zeros((1, image_size, image_size, 3))
            picname = filepath + class_list[i] + "/" + j
            img = misc.imread(picname)
            img = facenet.prewhiten(img)
            images[0, :, :, :] = img

            # sess.run(image_batch, {image_paths_placeholder: images})
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')


            prelogits, _ = network.inference(images_placeholder, keep_probability, phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size, weight_decay=weight_decay, reuse=False)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logits, inf_dict = sess.run([prelogits, _], feed_dict=feed_dict)
            # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            # print(embeddings)
            # k = embeddings.eval(session=sess)
            # print(k.shape, k)

