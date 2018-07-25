import tensorflow as tf
import cv2
import numpy as np
import os
from Models.architectures import *




class Discriminator(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 70, 70, 3], name='features')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Ck block
        self.C64 = Ck(self.input, 64, name="c64", with_batch_norm=False)
        self.C128 = Ck(self.C64, 128, name="c128")
        self.C256 = Ck(self.C128, 256, name="c256")
        self.C512 = Ck(self.C128, 512, name="c512")


    def step(self, session, images):
        outputs = session.run([self.C512, self.C256, self.C128, self.C64], {self.input.name: images})
        return outputs[0], outputs[1]


def readImages():
    folderName = r"D:\data\hackathon\profile2manga\Manga"
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