import tensorflow as tf
import cv2
import numpy as np
import os
from Models.architectures import *


class Generator(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='features')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        #TODO: check the reflection padding
        paddings = tf.constant([[2, 40], [3, 40]])
        self.padded_input = tf.pad(self.input, paddings, "REFLECT")

        # Convolution-BatchNorm-Relu encoder layers
        self.conv1 = Ck(self.padded_input, 32, name="conv1", stride=1, slope=0.5, is_training=self.is_training, kernel_size=9)
        self.conv2 = Ck(self.conv1, 64, name="conv2", stride=2, slope=0.5, is_training=self.is_training, kernel_size=3)
        self.conv3 = Ck(self.conv2, 128, name="conv3", stride=2, slope=0.5, is_training=self.is_training, kernel_size=3)

        # Residual blocks
        self.res_block1 = residual_block(self.conv3, alpha=0.2, name="res_block1", kernel_size=3, k=128, stride=1, slope=0.2, is_training=self.is_training)
        self.res_block2 = residual_block(self.res_block1, alpha=0.2, name="res_block2", kernel_size=3, k=128, stride=1, slope=0.2, is_training=self.is_training)
        self.res_block3 = residual_block(self.res_block2, alpha=0.2, name="res_block3", kernel_size=3, k=128, stride=1, slope=0.2, is_training=self.is_training)
        self.res_block4 = residual_block(self.res_block3, alpha=0.2, name="res_block4", kernel_size=3, k=128, stride=1, slope=0.2, is_training=self.is_training)
        self.res_block5 = residual_block(self.res_block4, alpha=0.2, name="res_block5", kernel_size=3, k=128, stride=1, slope=0.2, is_training=self.is_training)

        # Convolution-BatchNorm-Relu decoder layers
        self.decoder_conv1 = Ck(self.res_block5, 64, name="decoder_conv1", stride=0.5, slope=0.5, is_training=self.is_training, kernel_size=3)
        self.decoder_conv2 = Ck(self.decoder_conv1, 32, name="decoder_conv2", stride=0.5, slope=0.5, is_training=self.is_training, kernel_size=3)
        self.decoder_conv3 = Ck(self.decoder_conv2, 3, name="decoder_conv3", stride=1, slope=0.5, is_training=self.is_training, kernel_size=9)

