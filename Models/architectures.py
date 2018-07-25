import tensorflow as tf


""" Discriminator layers """
def Ck(input, k, name, stride=2, slope=0.2, is_training=True, kernel_size=4, with_batch_norm=True):
    """ Convolution-BatchNorm-LeakyRelu layers with k units.
    4x4 spatial convolution filters with stride 2, LeakyRelu with slope 0.2"""
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs=input,
                                filters=k,
                                kernel_size=[kernel_size, kernel_size],
                                strides=(stride, stride),
                                padding="same")
        batch_norm = tf.layers.batch_normalization(conv, training=is_training) if with_batch_norm else conv
        output = tf.nn.leaky_relu(batch_norm, alpha=slope)
        return output

def residual_block(input, alpha, name, kernel_size = 3, k=128, stride=1, slope=0.2, is_training=True):
    """ 3X3 Conv - BatchNorm - LeakyRelu - 3X3 Conv - BatchNorm - add input directly to output"""
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=input,
                                 filters=k,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=(stride, stride))

        batch_norm1 = tf.layers.batch_normalization(conv1, training=is_training)
        relu = tf.nn.leaky_relu(batch_norm1, alpha=slope)

        conv2 = tf.layers.conv2d(inputs=relu,
                                 filters=k,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=(stride, stride))

        batch_norm2 = tf.layers.batch_normalization(conv2, training=is_training)

        #TODO cut input appropriately and use matrix addition
        cut_side = kernel_size - 1
        output = batch_norm2 + alpha*(input[:, cut_side:-cut_side,cut_side:-cut_side, :])

        return output