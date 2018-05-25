class Config(object):
    def __init__(self):
        ## Questions and Annotataions JSON files
        self.DATA_DIR ='./datasets/'
        self.TRAIN_QUESTIONS_FILE='v2_OpenEnded_mscoco_train2014_questions.json'
        self.TRAIN_ANNOTATIONS_FILE='v2_mscoco_train2014_annotations.json'
        self.TRAIN_IMAGE_DIR = './tmp/'

        self.VAL_QUESTIONS_FILE='v2_OpenEnded_mscoco_val2014_questions.json'
        self.VAL_ANNOTATIONS_FILE='v2_mscoco_val2014_annotations.json'

        self.GLOVE_EMBEDDING_FILE='/Users/sainikhilmaram/Desktop/OneDrive/UCSB courses/Spring_2018/click-bait/Multimodal-Clickbait-Detection/glove.6B.100d.txt'

        ## CNN parameters
        self.TRAIN_CNN = False
        self.CNN='vgg16'
        self.CNN_PRETRAINED_FILE = self.DATA_DIR +'./vgg16_weights.npz'
        self.IMAGE_SHAPE = [224, 224, 3]

        # self.CNN = 'resnet50'
        # self.CNN_PRETRAINED_FILE = './resnet50_no_fc.npy'

        ## RNN PARAMETERS
        self.MAX_QUESTION_LENGTH = 25
        self.EMBEDDING_DIMENSION = 100



        ## Decoder Parameters
        self.OUTPUT_SIZE = self.TOP_ANSWERS
        self.TOP_ANSWERS = 1000
        self.ONLY_TOP_ANSWERS = True ## If we are considering only questions with top answers in our model
        self.MAX_ANSWER_LENGTH = 1


        ## Model Parameters
        self.PHASE = 'train'
        self.POINT_WISE_FEATURES = 1024
        self.BATCH_SIZE = 10
        self.INITIAL_LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 1
        self.NUM_BATCHES = 20 ## Just a place holder, so it doesn't loop through all the data.

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01
