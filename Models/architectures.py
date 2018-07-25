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

def residual_block(input, alpha, kernel_size = 3, k=128, stride=1, slope=0.2, name=None, is_training=True):
    """ 3X3 Conv - BatchNorm - LeakyRelu - 3X3 Conv - BatchNorm - add input directly to output"""
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=input,
                                filters=k,
                                kernel_size=[kernel_size,kernel_size],
                                strides=stride,
                                padding="valid")

        batch_norm1 = tf.layers.batch_normalization(conv1, training=is_training)
        relu = tf.nn.leaky_relu(batch_norm1, alpha=slope)

        conv2 = tf.layers.conv2d(inputs=relu,
                                 filters=k,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=stride,
                                 padding="valid")

        batch_norm2 = tf.layers.batch_normalization(conv2, training=is_training)

        #TODO cut input appropriately and use matrix addition
        cut_side = (kernel_size - 1) / 2
        output = batch_norm2 + alpha*(input[:,:,cut_side:-cut_side,cut_side:-cut_side])

        return output