""""
This encoder model consists of instances of cnn and LSTM.
Here we combine output of each model to a separate fully connected layer to get the outputs of 1024.
In Hierachical co-attention model, this encoder model consists of code where co-attention is possible

"""
import tensorflow as tf
from vqa_cnn import *

class vqa_encoder:
    def __init__(self,config):
        self.config = config
        self.cnn = vqa_cnn(self.config)
        ## LSTM code here

    def build(self,images,questions):
        ## Build the CNN model
        self.cnn.build(images)
        ## Build the sentence level RNN Model

        ## Combine the model
        self.build_encoder()


    def build_encoder(self):

        ## Get the features from rnn model

        ## Build a fully connected layer for CNN and RNN to get 1024 features for each
        ## Get the features from CNN model
        with tf.variable_scope('fc_cnn_model', reuse=tf.AUTO_REUSE) as scope:
            fc_cnn_model_w = tf.get_variable(
                initializer=tf.truncated_normal([self.cnn.dim_ctx, self.config.POINT_WISE_FEATURES],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='fc_W', trainable=True)
            fc_cnn_model_b = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.POINT_WISE_FEATURES], dtype=tf.float32),
                                  trainable=True, name='fc_b')
            self.cnn_features = tf.nn.bias_add(tf.matmul(self.cnn.conv_feats, fc_cnn_model_w), fc_cnn_model_b)

        ## In the same fashion build the RNN model too
        self.rnn_features = tf.Variable([self.config.BATCH_SIZE, self.config.POINT_WISE_FEATURES])


