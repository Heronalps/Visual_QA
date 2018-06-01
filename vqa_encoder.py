""""
This encoder model consists of instances of cnn and LSTM.
Here we combine output of each model to a separate fully connected layer to get the outputs of 1024.
In Hierachical co-attention model, this encoder model consists of code where co-attention is possible

"""
import tensorflow as tf
from vqa_cnn import *
from vqa_lstm import *

class vqa_encoder:
    def __init__(self,config):
        self.config = config
        self.cnn = vqa_cnn(self.config)
        ## LSTM code here
        self.lstm = vqa_lstm(self.config)

    #def build(self,images,questions,question_masks,embedding_matrix):
    def build(self, image_features, questions, question_masks, embedding_matrix):
        if self.config.PHASE == "test":
            ## Build the CNN model
            images = image_features
            self.cnn.build(images)
        self.image_features = image_features
        ## Build the sentence level LSTM Model

        self.lstm.build(questions,question_masks,embedding_matrix)
        ## Combine the model
        self.build_encoder()


    def build_encoder(self):
        ## Get the features from CNN model
        if self.config.PHASE == "test":
            self.conv_features = self.cnn.conv_feats
        else:
            self.conv_features = self.image_features

        print("CNN feature size {}".format(self.conv_features.get_shape()))
        with tf.variable_scope('fc_cnn_model', reuse=tf.AUTO_REUSE) as scope:
            fc_cnn_model_w = tf.get_variable(
                initializer=tf.truncated_normal([self.config.IMAGE_FEATURES_MAP, self.config.POINT_WISE_FEATURES],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='W', trainable=True)
            fc_cnn_model_b = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.POINT_WISE_FEATURES], dtype=tf.float32),
                                  trainable=True, name='B')
            self.cnn_features = tf.nn.bias_add(tf.matmul(self.conv_features, fc_cnn_model_w), fc_cnn_model_b)
        self.cnn_features = tf.nn.relu(self.cnn_features)

        ## Get the features from LSTM model
        with tf.variable_scope('fc_lstm_model', reuse=tf.AUTO_REUSE) as scope:
            fc_lstm_model_w = tf.get_variable(
                initializer=tf.truncated_normal([self.lstm.dim, self.config.POINT_WISE_FEATURES],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='W', trainable=True)
            fc_lstm_model_b = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.POINT_WISE_FEATURES], dtype=tf.float32),
                                  trainable=True, name='B')
            self.lstm_features = tf.nn.bias_add(tf.matmul(self.lstm.lstm_features, fc_lstm_model_w), fc_lstm_model_b)
        self.lstm_features = tf.nn.relu(self.lstm_features)




