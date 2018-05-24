"""
VQA LSTM part
"""
import tensorflow as tf

# BATCH_START = 0
# TIME_STEPS = 20
# BATCH_SIZE = 50
# INPUT_SIZE = 25
# OUTPUT_SIZE = 1024
# CELL_SIZE = 10
# LR = 0.006


class vqa_lstm(object):
    def __init__(self, config):
        self.n_steps = config.LSTM_STEPS
        self.input_size = config.LSTM_INPUT_SIZE
        self.output_size = config.LSTM_OUTPUT_SIZE
        self.cell_size = config.LSTM_CELL_SIZE
        self.batch_size = config.LSTM_BATCH_SIZE
        self.learning_rate = config.LSTM_LEARN_RATE
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, self.n_steps, self.output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()

        ## Training part is in the vqa_main ##

        # with tf.variable_scope('out_hidden'):
        #     self.add_output_layer()
        # with tf.name_scope('cost'):
        #     self.compute_cost()
        # with tf.name_scope('train'):
        #     self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, input_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)


    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def encode(self, dataset):
        sess = tf.Session()

        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("logs", sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
        # $ tensorboard --logdir='logs'

        # this part should go to vqa_main
        count = 0
        while dataset.has_next_batch():
            _, question_idxs, question_masks, _, _ = dataset.next_batch()
            if count == 0:
                feed_dict = {
                    self.xs: question_idxs,
                    # create initial state
                }
            else:
                feed_dict = {
                    self.xs: question_idxs,
                    self.cell_init_state: state  # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [self.train_op, self.cost, self.cell_final_state, self.pred],
                feed_dict=feed_dict)

            # if i % 20 == 0:
            #     print('cost: ', round(cost, 4))
            #     result = sess.run(merged, feed_dict)
            #     writer.add_summary(result, i)
