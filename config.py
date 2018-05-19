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
        self.CNN_PRETRAINED_FILE='./vgg16_weights.npz'
        # self.CNN = 'resnet50'
        # self.CNN_PRETRAINED_FILE = './resnet50_no_fc.npy'

        ## PARAMETERS
        self.MAX_QUESTION_LENGTH = 25
        self.MAX_ANSWER_LENGTH = 25
        self.BATCH_SIZE = 10

        self.PHASE = 'train'


        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01
