import tensorflow as tf
import numpy as np
from vqa_preprocessing import  *
from config import *
from vqa_vocabulary import *
import argparse
import sys

from vqa_cnn import *




def loadGlove(embeddingFile):
    vocab = []
    embedding = []
    dictionary = {}
    reverseDictionary = {}
    count = 0
    print("Loading Glove")
    file = open(embeddingFile, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embedding.append(row[1:])
        dictionary[row[0]] = count
        reverseDictionary[count] = row[0]
        count = count + 1
    print('Loaded GloVe!')
    file.close()
    print(len(vocab))
    return vocab, embedding,dictionary,reverseDictionary

def parse_args(args):
    """
    Parse the arguments from the given command line
    Args:
        args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
    """

    parser = argparse.ArgumentParser()


    # Dataset options
    datasetArgs = parser.add_argument_group('Dataset options')

    datasetArgs.add_argument('--data_dir', type=str, default='./datasets/', help='Data directory')

    datasetArgs.add_argument('--train_questions_file', type=str, default='v2_OpenEnded_mscoco_train2014_questions.json', help='training questions data set')
    datasetArgs.add_argument('--train_annotations_file', type=str, default='v2_mscoco_train2014_annotations.json',help='training annotations data set')

    datasetArgs.add_argument('--val_questions_file', type=str, default='v2_OpenEnded_mscoco_val2014_questions.json', help='validation questions data set')
    datasetArgs.add_argument('--val_annotations_file', type=str, default='v2_mscoco_val2014_annotations.json',help='validation annotations data set')


    # # Network options (Warning: if modifying something here, also make the change on save/loadParams() )
    # nnArgs = parser.add_argument_group('Network options', 'architecture related option')
    # nnArgs.add_argument('--hiddenSize', type=int, default=512, help='number of hidden units in each RNN cell')
    # nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
    # nnArgs.add_argument('--softmaxSamples', type=int, default=0, help='Number of samples in the sampled softmax loss function. A value of 0 deactivates sampled softmax')
    # nnArgs.add_argument('--initEmbeddings', action='store_true', help='if present, the program will initialize the embeddings with pre-trained word2vec vectors')
    # nnArgs.add_argument('--embeddingSize', type=int, default=64, help='embedding size of the word representation')
    # nnArgs.add_argument('--embeddingSource', type=str, default="GoogleNews-vectors-negative300.bin", help='embedding file to use for the word representation')

    ## cnn options
    cnnArgs = parser.add_argument_group('CNN options')
    cnnArgs.add_argument('--cnn',type=str,default='vgg16',help='vgg model to be loaded')
    cnnArgs.add_argument('--cnn_pretrained_file',type=str,default='./datasets/vgg16_weights.npz',help='pretrained vgg model')
    cnnArgs.add_argument('--train_cnn',type=bool,default=False,help='To update the weights of CNN using training')
    # Training options
    trainingArgs = parser.add_argument_group('Training options')
    trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
    trainingArgs.add_argument('--saveEvery', type=int, default=2000, help='nb of mini-batch step before creating a model checkpoint')
    trainingArgs.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    trainingArgs.add_argument('--learningRate', type=float, default=0.002, help='Learning rate')
    trainingArgs.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probabilities)')

    trainingArgs.add_argument('--max_question_length', type=int, default=25, help='maximum question length')
    trainingArgs.add_argument('--max_answer_length', type=int, default=1, help='maximum answer length')



    return parser.parse_args(args)

def assign_args(args):
    config = Config()
    ## Update config parameters with the ones passed from command line
    config.DATA_DIR = args.data_dir
    config.TRAIN_QUESTIONS_FILE = args.train_questions_file
    config.TRAIN_ANNOTATIONS_FILE = args.train_annotations_file
    config.VAL_QUESTIONS_FILE = args.val_questions_file
    config.VAL_ANNOTATIONS_FILE = args.val_annotations_file
    config.MAX_QUESTION_LENGTH=args.max_question_length
    config.MAX_ANSWER_LENGTH = args.max_answer_length
    config.BATCH_SIZE = args.batch_size

    ## CNN
    config.CNN = args.cnn
    config.CNN_PRETRAINED_FILE = args.cnn_pretrained_file
    config.TRAIN_CNN = args.train_cnn
    return config

if __name__ == "__main__":
    # ## Use the arguments from the command line in the fina model
    # args = sys.argv[1:]
    # ## Parse the input arguments
    # parsed_args = parse_args(args)
    #
    # ## assign input arguments to the config object
    # config = assign_args(parsed_args)

    config = Config()

    vocab,embedding,dictionary,reverseDictionary = loadGlove(config.GLOVE_EMBEDDING_FILE)
    data_set = prepare_train_data(config,vocab,dictionary)

    for i in range(100):
        print(data_set.answer_idxs_list[i],data_set.answer_masks_list[i])


    # sess = tf.Session()
    #
    # model = vqa_cnn(config)
    # model.build()
    # model.load_cnn(sess,config.CNN_PRETRAINED_FILE)







