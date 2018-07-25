import tensorflow as tf
import cv2
import numpy as np
import os
from Models.architectures import *




class Discriminator(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='features')

        # Ck block
        self.C128 = Ck(self.input, 128, name="c128")

        # res block 1
        self.res_block1 = residual_block(self.C128, alpha=0.5, name="res_block")


    def step(self, session, images):
        outputs = session.run([self.res_block1, self.C128], {self.input.name: images})
        return outputs[0], outputs[1]


def readImages():
    folderName = r"C:\Users\Michal\Documents\Hackaton2018 - Datasets\Manga faces\Manga"
    folder_images = [folderName + "/"+ f for f in os.listdir(folderName)]
    n = 5
    image_list = []
    for im_name in folder_images[:n]:
        im = cv2.imread(im_name)
        image_list.append(im)
    image_array = np.array(image_list)
    return image_array


def create_model(sess, model_path):
    """
    Creates the model object with its graph, if there is an existing checkpoint restore the models variables,
    otherwise initialize them.
    :param sess: tensorflow session
    :return: the model
    """
    model = Discriminator()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    return model

def testNet():
    imagesArray = readImages()
    with tf.Session() as sess:
        testmodel = create_model(sess, " ")
        res_output, ck_output = testmodel.step(sess, imagesArray)
        print("done")

if __name__ == "__main__":
    testNet()