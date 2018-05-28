"""
This is the model file which has instance of encoder an decoder
functions for training,testing and evaluation of model

"""

import tensorflow as tf

from vqa_encoder import *
from vqa_decoder import *
from vqa_preprocessing import *

class vqa_model:
    def __init__(self,config):
        self.config = config
        self.encoder = vqa_encoder(self.config)
        self.decoder = vqa_decoder(self.config)
        self.image_loader = ImageLoader('./ilsvrc_2012_mean.npy')

    def build(self):
        ## Build the encoder and decoder models
        ## Place holder fo the images and questions and we pass them to the encoder
        self.images = tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + self.config.image_shape)
        self.questions =tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + [self.config.MAX_QUESTION_LENGTH]+[self.config.EMBEDDING_DIMENSION])
        self.question_masks = tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + [self.config.MAX_QUESTION_LENGTH] + [self.config.EMBEDDING_DIMENSION])


        self.embedding_matrix_placeholder = tf.placeholder(tf.float32, shape=[self.config.VOCAB_SIZE, self.config.EMBEDDING_DIMENSION])

        self.embedding_matrix = tf.Variable(tf.constant(0.0, shape=[self.config.VOCAB_SIZE, self.config.EMBEDDING_DIMENSION]),
                        trainable=False, name="embedding_matrix")

        ## pass the images, questions and embedding matrix to the encoder
        self.encoder.build(self.images,self.questions,self.question_masks, self.embedding_matrix)
        ## pass the outputs of encoder to decoder model
        self.decoder.build(self.encoder.cnn_features,self.encoder.lstm_features)

        self.build_model()

    def build_model(self):
        ## Assign variables that needs to be passed to variables from encoder and decoder
        pass

    def train(self,sess,train_data,embedding_matrix_glove):
        print("Training the model")

        ## Assign embedding matrix to the variable in session
        self.embedding_init = self.embedding_matrix.assign(self.embedding_matrix_placeholder)

        sess.run(self.embedding_init, feed_dict={self.embedding_matrix_placeholder: embedding_matrix_glove})

        for _ in tqdm(list(range(self.config.NUM_EPOCHS)), desc='epoch'):
            #for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
            for _ in tqdm(list(range(self.config.NUM_BATCHES)), desc='batch'):
                batch = train_data.next_batch()
                image_files, question_idxs, question_masks, answer_idxs, answer_masks = batch
                images = self.image_loader.load_images(image_files)

                feed_dict = {self.images:images,
                             self.questions:question_idxs,
                             self.question_masks:question_masks,
                             self.decoder.answers:answer_idxs,
                             self.decoder.answer_masks:answer_masks}

                _ = sess.run(self.decoder.optimizer,feed_dict=feed_dict)





