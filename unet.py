# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
import tensorflow as tf
from tensorflow import concat, variable_scope
from functools import partial


xavier_init = tf.contrib.layers.xavier_initializer
conv2d = partial(tf.layers.conv2d,
                 activation=tf.nn.relu,
                 kernel_initializer=xavier_init(),
                 padding="SAME")
conv2d_linear = partial(tf.layers.conv2d,
                        kernel_initializer=xavier_init(),
                        padding="SAME")
UpSampling2D = tf.keras.layers.UpSampling2D
max_pooling2d = tf.layers.max_pooling2d
dropout = partial(tf.layers.dropout)
batch_norm = tf.layers.batch_normalization

KERNEL_SIZE = 5


def downsample(in_data, filters, last=False, name='DownSample', reuse=False):
    """
    Down-Sample residual of U-Net.
    :param in_data: The data to down-sample.
    :param filters: The filter size for each convolution.
    :param last: For the last down-sample - instead of
                   cropping and pooling, do dropout.
    :param name: The name for the down-sample residual.
    :param reuse: Reuse weights or not.
    :return: The residual output.
    """
    with variable_scope(name, reuse=reuse):
        conv1 = conv2d(in_data, filters, KERNEL_SIZE)
        conv2 = conv2d(conv1, filters, KERNEL_SIZE)
        bn = batch_norm(conv2)
        if last:
            return dropout(bn)
        return max_pooling2d(bn, 2, 2), conv1


def upsample(in_data, crop, filters, name='UpSample', reuse=False):
    """
    Up-Sample residual of U-Net.
    :param in_data: The data to up-sample.
    :param crop: The cropping to connect
    :param filters: The filter size for each convolution.
    :param name: The name for the down-sample residual.
    :param reuse: Reuse weights or not.
    :return: The residual output.
    """
    with variable_scope(name, reuse=reuse):
        up = UpSampling2D(size=(2, 2))(in_data)
        conv1 = conv2d(up, filters, KERNEL_SIZE, padding="SAME")
        merge = concat([crop, conv1], axis=3)
        conv2 = conv2d(merge, filters, KERNEL_SIZE)
        conv3 = conv2d(conv2, filters, KERNEL_SIZE)
        drop2 = dropout(conv3)
        return drop2


def unet(in_data, name='UNet', reuse=False):
    """
    Define Unet, you can refer to:
     - http://blog.csdn.net/u014722627/article/details/60883185
     - https://github.com/zhixuhao/unet
    :param in_data: Input data.
    :param name: Name for the unet residual
    :param reuse: Reuse weights or not.
    :return: The result of the last layer of U-Net.
    """
    assert in_data is not None

    with variable_scope(name, reuse=reuse):
        pool1, res1 = downsample(in_data, 32, name='DownSample1', reuse=reuse)
        pool2, res2 = downsample(pool1, 64, name='DownSample2', reuse=reuse)
        pool3, res3 = downsample(pool2, 128, name='DownSample3', reuse=reuse)
        bridge = downsample(pool3, 256, name='Bridge', reuse=reuse, last=True)
        up3 = upsample(bridge, res3, 128, name='UpSample3', reuse=reuse)
        up2 = upsample(up3, res2, 64, name='UpSample2', reuse=reuse)
        up1 = upsample(up2, res1, 32, name='UpSample1', reuse=reuse)
        conv = conv2d(up1, KERNEL_SIZE, 3, padding="SAME", name='Conv')
        out = conv2d_linear(conv, 3, 1, name='Out')
    return out
