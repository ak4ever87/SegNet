"""
Fusion-Net Implementation based on: https://arxiv.org/abs/1711.07341 with some modifications and our adaptations.
"""

from __future__ import absolute_import, division
import tensorflow as tf
from tensorflow import variable_scope
from functools import partial
from utils import bn_conv_layer, bn_conv2d_transpose_layer, conv2d_linear

UpSampling2D = tf.keras.layers.UpSampling2D
max_pooling2d = tf.layers.max_pooling2d
dropout = partial(tf.layers.dropout, rate=0.3)


class FusionNet(object):
    def __init__(self, layers, kernel_size, kernel_num, num_repeat):
        """
        Create the fusion-net model instance.
        :param layers: The number of down and up layers.
        :param kernel_size: The size of the kernel window.
        :param kernel_num: The size of the initial resulting kernel count which grow by two on each depth.
        :param num_repeat: Number of internal convolutions between the resnet connection.
        """
        self.layers = layers
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.num_repeat = num_repeat

    def conv_res_conv_block(self, prev_layer, output_dim,
                            name="ConvResConvBlock", training=True):
        """
        Resnet block.
        :param prev_layer: Previous incomming layer.
        :param output_dim: The dimension of the each convolusion kernel dimension.

        :param name: The name for this layer.
        :param training: Whether or not it is a train run.
        """
        with variable_scope(name):
            conv1 = bn_conv_layer(prev_layer, output_dim, self.kernel_size, name="Conv1", training=training)
            output = conv1
            for i in range(self.num_repeat):
                conv = bn_conv_layer(output, output_dim, self.kernel_size, name="conv_{0}".format(i), training=training)
                output = conv
            res = conv1 + output
            conv2 = bn_conv_layer(res, output_dim, self.kernel_size, name="Conv2", training=training)
            return conv2

    def encoder(self, prev_layer, name='Encoder', training=True):
        """
        The encoding layer block.
        :param prev_layer: Previous incomming layer.
        :param name: The name for this layer.
        :param training: Whether or not it is a train run.
        """
        with variable_scope(name):
            pool = prev_layer
            skip_links = []
            for i in range(self.layers):
                down = self.conv_res_conv_block(pool, self.kernel_num * (2 ** i),
                                                name="DownSample{0}".format(i + 1),
                                                training=training)
                bn = down
                pool = max_pooling2d(bn, 2, 2, name="Pool{0}".format(i + 1))
                skip_links.insert(0, down)
            return pool, skip_links

    def decoder(self, prev_layer, skip_links, name='Decoder', training=True):
        """
        The decoding block.
        :param prev_layer: The encoded output.
        :param skip_links: The skip_links saved by the encoder block.
        :param name: The name for this layer.
        :param training: Whether or not it is a train run.
        """
        with variable_scope(name):
            d = prev_layer
            l = len(skip_links)
            for i, link in enumerate(skip_links):
                cur_kernel_num = self.kernel_num * (2 ** (l - i - 1))
                bn = d
                res = 0.5 * (bn_conv2d_transpose_layer(bn, cur_kernel_num, self.kernel_size, strides=2,
                                                       name="Res{0}".format(l - i), training=training) + link)
                up = self.conv_res_conv_block(res, cur_kernel_num, name="UpSample{0}".format(l - i),
                                              training=training)
                d = dropout(up, name="Dropout{0}".format(l - i), training=training)
        return d

    def fusion_net(self, input_data, training):
        """
        Create the fusion-net model.
        :param input_data: Input to the layer.
        :param training: Whether or not it is a train run.
        :return The model.
        """
        encode_vec, skip_links = self.encoder(input_data, training=training)
        boost = 2 ** self.layers
        # boost = 1
        bridge = self.conv_res_conv_block(encode_vec, self.kernel_num * boost,
                                          name="Bridge", training=training)
        d = dropout(bridge, training=training)
        decode_vec = self.decoder(d, skip_links, training=training)
        output = conv2d_linear(decode_vec, 3, self.kernel_size, name="Output")
        return output
