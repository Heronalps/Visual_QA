"""
VQA LSTM part
"""
import tensorflow as tf
import numpy as np

class vqa_lstm(object):
    def __init__(self, config):
        self.n_steps = config.LSTM_STEPS
        self.input_size = config.LSTM_INPUT_SIZE
        self.output_size = config.LSTM_OUTPUT_SIZE
        self.cell_size = config.LSTM_CELL_SIZE
        self.batch_size = config.LSTM_BATCH_SIZE
        self.lstm_layer = 2
        self.dim = self.lstm_layer * 1024

    def build(self, question_idxs, questions_mask, embedding_matrix):

        word_embed = tf.nn.embedding_lookup(embedding_matrix, question_idxs)

        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
        multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(cells = [lstm_cell_1, lstm_cell_2])

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell_1.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            multi_lstm_cell, word_embed, initial_state=self.cell_init_state, time_major=False)

        self.lstm_features = tf.concat([self.cell_final_state[-1][0], self.cell_final_state[-1][1]], 0)

