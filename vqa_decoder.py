import tensorflow as tf
import numpy as np

class vqa_decoder:
    def __init__(self,config,image_features,question_features):
        self.config =  config
        self.image_features = image_features
        self.question_features = question_features

    def build_decoder(self):
        ## Point wise multiplication
        ## Check if both the shapes are matching
        config = self.config
        self.is_train = True

        # Setup the placeholders
        if self.is_train:
            # contexts = self.conv_feats
            self.answers = tf.placeholder(
                dtype=tf.int32,
                shape=[config.BATCH_SIZE, config.MAX_ANSWER_LENGTH])
            self.masks = tf.placeholder(
                dtype=tf.float32,
                shape=[config.BATCH_SIZE, config.MAX_ANSWER_LENGTH])

        if(tf.shape(self.image_features)[1] == config.POINT_WISE_FEATURES and
           tf.shape(self.question_features)[1] == config.POINT_WISE_FEATURES):
            self.point_wise = tf.multiply(self.image_features,self.question_features)

        else:
            ## Reshape them into correct shape
            self.image_features = tf.reshape(self.image_features,[config.BATCH_SIZE,config.POINT_WISE_FEATURES])
            self.question_features = tf.reshape(self.question_features,[config.BATCH_SIZE,config.POINT_WISE_FEATURES])

            self.point_wise = tf.multiply(self.image_features,self.question_features)


        ## Build a Fully Connected Layer
        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
            fcw = tf.get_variable(initializer=tf.truncated_normal([self.config.POINT_WISE_FEATURES, self.config.OUTPUT_SIZE],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='fc_W',trainable=True)
            fcb = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.OUTPUT_SIZE], dtype=tf.float32),
                               trainable=True, name='fc_b')
            fcl = tf.nn.bias_add(tf.matmul(self.point_wise, fcw), fcb)
            logits = tf.nn.relu(fcl)

        if self.is_train:
            # Compute the loss for this step, if necessary
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.onehot_encode(self.onehot_encode(self.answers[:,0])),
                logits=logits)

            self.optimizer = tf.train.AdamOptimizer(config.INITIAL_LEARNING_RATE).minimize(cross_entropy_loss)

        self.predictions = tf.argmax(logits, 1)


    def onehot_encode(self,answers):
        onehot_vector = []
        for ans in answers:
            vector = np.zeros(self.config.TOP_ANSWERS)
            vector[int(ans)] = 1
            onehot_vector.append(vector)
        return np.array(onehot_vector)