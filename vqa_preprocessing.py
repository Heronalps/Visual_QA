## preprocessing training data and storing it in data object
## preprocessing test data and storing it in data object
## preprocessing validation data and storing it in data object.

"""
Preporcessing contains following steps
1. Remove unnecessary words
2. Load the glove
3.convert the words to indices in the vocabulary
4. Mask to the maximum length.
"""

import json
import numpy as np
import os
from vqa_vocabulary import Vocabulary
from tqdm import tqdm
from vqa_dataset import DataSet



# val_annot = json.load(open(config.DATA_DIR+config.VALIDATION_ANNOTATION_FILE, 'r'))
#
# val_ques = json.load(open(config.DATA_DIR+config.VALIDATION_QUESTION_FILE, 'r'))

def get_best_confidence_answer(answer_list):
    """Gives the best confidence answer from answer list"""
    best_confidence = 0.0
    best_answer =""
    best_answer_id = 0
    ##Currently just checking if answer confidence is yes. Need to pick a better answer//TODO
    for answer in answer_list:
        if(answer['answer_confidence'] == "yes"):
            best_confidence = answer['answer_confidence']
            best_answer = answer['answer']
            best_answer_id = answer['answer_id']
            break

    return best_answer,best_answer_id


def get_top_answers(config):
    """TO get the top answers as we are classifying the answers into one of top 1000"""
    counts = {}
    train_annot = json.load(open(config.DATA_DIR + config.TRAIN_ANNOTATIONS_FILE, 'r'))
    train_size = len(train_annot['annotations'])
    for i in tqdm(list(range(train_size)), desc='Answers'):
        ## Get the top 1000 best confidence answers
        answer, answer_id = get_best_confidence_answer(train_annot['annotations'][i]['answers'])
        counts[answer] =counts.get(answer,0) + 1

    count_words = sorted([(count, w) for w, count in counts.items()], reverse=True)

    # print('top answer and their counts:')
    # print('\n'.join(map(str, count_words[:20])))

    count = 0
    for i in range(config.TOP_ANSWERS):
        count += count_words[i][0]

    print("Total data set with top {0} answers : {1}".format(config.TOP_ANSWERS,count))

    top_answers = []
    for i in range(config.TOP_ANSWERS):
        top_answers.append(count_words[i][1])

    return top_answers


def prepare_train_data(config,words,word2idx):
    """ Prepare the data for training the model. """
    print("Loading the training json files")
    train_annot = json.load(open(config.DATA_DIR + config.TRAIN_ANNOTATIONS_FILE, 'r'))
    train_ques = json.load(open(config.DATA_DIR + config.TRAIN_QUESTIONS_FILE, 'r'))

    train_size = len(train_annot['annotations'])
    vocabulary = Vocabulary(words,word2idx)

    ## Lists that need to be passed to DataSet object
    image_id_list = [] ; image_file_list = []
    question_id_list = [] ; question_idxs_list = [] ; question_masks_list = [] ; question_type_list = []
    answer_id_list = []  ; answer_idxs_list = [] ; answer_masks_list = [] ; answer_type_list = []

    print("Preparing Training Data...")

    top_answers = get_top_answers(config)
    answer_to_idx = {ans:idx for idx,ans in enumerate(top_answers)}
    idx_to_answer = {idx:ans for idx,ans in enumerate(top_answers)}

    for i in tqdm(list(range(train_size)), desc='training data'):
        ## Attributes required from questions
        question_id = train_ques['questions'][i]['question_id']
        image_id    = train_ques['questions'][i]['image_id']
        question    = train_ques['questions'][i]['question']
        image_file  = os.path.join(config.TRAIN_IMAGE_DIR,str(image_id))

        ## Attributes required from annotations
        question_type = train_annot['annotations'][i]['question_type']
        answer_type   = train_annot['annotations'][i]['answer_type']
        answer,answer_id = get_best_confidence_answer(train_annot['annotations'][i]['answers'])

        ## config.ONLY_TOP_ANSWERS, then the answer_idxs will contain the answer index in the top answers
        if config.ONLY_TOP_ANSWERS:
            if answer in top_answers:
                ## Convert question into question indexes
                question_idxs_ = vocabulary.process_sentence(question)
                question_num_words = len(question_idxs_)

                question_idxs = np.zeros(config.MAX_QUESTION_LENGTH,dtype = np.int32)
                question_masks = np.zeros(config.MAX_QUESTION_LENGTH)

                question_idxs[:question_num_words] = np.array(question_idxs_)
                question_masks[:question_num_words] = 1


                ## Convert the answer into answer indexes
                answer_idxs_ = answer_to_idx[answer]
                answer_num_words = 1

                answer_idxs = np.zeros(config.MAX_ANSWER_LENGTH, dtype=np.int32)
                answer_masks = np.zeros(config.MAX_ANSWER_LENGTH)

                answer_idxs[:answer_num_words] = np.array(answer_idxs_)
                answer_masks[:answer_num_words] = 1

                ## Place the elements into their list
                image_id_list.append(image_id) ; image_file_list.append(image_file)

                question_id_list.append(question_id) ; question_idxs_list.append(question_idxs)
                question_masks_list.append(question_masks) ; question_type_list.append(question_type)

                answer_id_list.append(answer_id);  answer_idxs_list.append(answer_idxs)
                answer_masks_list.append(answer_masks); answer_type_list.append(answer_type)

        else:
            ## This is used in future if we are planning to have decoder as an LSTM unit
            ## Convert question into question indexes
            question_idxs_ = vocabulary.process_sentence(question)
            question_num_words = len(question_idxs_)

            question_idxs = np.zeros(config.MAX_QUESTION_LENGTH, dtype=np.int32)
            question_masks = np.zeros(config.MAX_QUESTION_LENGTH)

            question_idxs[:question_num_words] = np.array(question_idxs_)
            question_masks[:question_num_words] = 1

            ## Convert the answer into answer indexes
            answer_idxs_ = vocabulary.process_sentence(answer)
            answer_num_words = len(answer_idxs_)

            answer_idxs = np.zeros(config.MAX_ANSWER_LENGTH, dtype=np.int32)
            answer_masks = np.zeros(config.MAX_ANSWER_LENGTH)

            answer_idxs[:answer_num_words] = np.array(answer_idxs_)
            answer_masks[:answer_num_words] = 1

            ## Place the elements into their list
            image_id_list.append(image_id);
            image_file_list.append(image_file)

            question_id_list.append(question_id);
            question_idxs_list.append(question_idxs)
            question_masks_list.append(question_masks);
            question_type_list.append(question_type)

            answer_id_list.append(answer_id);
            answer_idxs_list.append(answer_idxs)
            answer_masks_list.append(answer_masks);
            answer_type_list.append(answer_type)


    image_id_list = np.array(image_id_list) ; image_file_list = np.array(image_file_list)

    question_id_list = np.array(question_id_list) ;  question_idxs_list = np.array(question_idxs_list)
    question_masks_list = np.array(question_masks_list) ; question_type_list = np.array(question_type_list)

    answer_id_list = np.array(answer_id_list) ; answer_idxs_list = np.array(answer_idxs_list)
    answer_masks_list = np.array(answer_masks_list) ; answer_type_list = np.array(answer_type_list)

    #print(image_id_list,question_id_list,question_idxs_list,answer_idxs_list)



    print("Number of Questions = %d" %(train_size))
    print("Missing words : ", vocabulary.missingWords)
    print("Building the dataset...")
    dataset = DataSet(image_id_list,
                      image_file_list,
                      question_id_list,
                      question_idxs_list,
                      question_masks_list,
                      question_type_list,
                      answer_id_list,
                      answer_idxs_list,
                      answer_masks_list,
                      answer_type_list,
                      config.BATCH_SIZE,
                      True,
                      True)
    print("Dataset built.")
    return dataset
