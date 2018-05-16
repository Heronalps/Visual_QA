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

        ## PARAMETERS
        self.MAX_QUESTION_LENGTH = 25
        self.MAX_ANSWER_LENGTH = 25
        self.BATCH_SIZE = 10
