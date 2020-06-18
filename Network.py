import tensorflow as tf
from fusion_net import FusionNet


class Network(object):
    def __init__(self, **params):
        self.net = FusionNet(**params)

    def build(self, input_batch, training=True):
        """
        This function is where you write the code for your network.
          The input is a batch of images of size (N,H,W,1)
        N is the batch size
        H is the image height
        W is the image width
        The output needs to be of shape (N,H,W,3)
        where the last channel is the UNNORMALIZEd calss probabilities
          (before softmax) for classes background,
        foreground and edge.
        :param input_batch: The batch for the current train.
        :param training: Whether or not it is a train run.
        :return: The output from the network.
        """
        out = self.net.fusion_net(input_batch, training=training)
        tf.summary.scalar('out-mean', tf.reduce_mean(out))
        return out
