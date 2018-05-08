"""
Download VQA dataset and preprocessing

05/07/2018 Version 1.0
"""

# Download the VQA train + validation + test dataset from http://visualqa.org/download.html

import json
import os
import argparse

def download_vqa():
    # Download VQA Questions
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip -P zip/')

    # Download VQA Annotations
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/')

    # Unzip datasets
    os.system('wget zip/v2_v2_Questions_Train_mscoco.zip -d vqa/')
    os.system('wget zip/v2_v2_Questions_Val_mscoco.zip -d vqa/')
    os.system('wget zip/v2_v2_Questions_Test_mscoco.zip -d vqa/')
    os.system('wget zip/v2_v2_Annotations_Train_mscoco.zip -d vqa/')
    os.system('wget zip/v2_v2_Annotations_Val_mscoco.zip -d vqa/')

def main(param):
    

