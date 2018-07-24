import tensorflow as tf


""" Discriminator layers """
def Ck(input, k, stride=2, slope=0.2, name=None, is_training=True):
    """ Convolution-BatchNorm-LeakyRelu layers with k units.
    4x4 spatial convolution filters with stride 2, LeakyRelu with slope 0.2"""
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=input,
                                filters=k,
                                kernel_size=[4,4],
                                strides=stride,
                                padding="same")
        batch_norm = tf.contrib.layers.batch_norm(conv,
                                                  decay=0.9,
                                                  scale=True,
                                                  updates_collections=None,
                                                  is_training=is_training)
        output = tf.nn.leaky_relu(batch_norm, alpha=slope)
        return output