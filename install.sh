#!/usr/bin/env bash

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

jar xf train2014.zip
jar xf val2014.zip
jar xf test2015.zip

pip3 install --upgrade pip

pip install tensorflow

pip install pandas

pip install tqdm

pip install nltk

pip install opencv-python

pip install scipy

pip install Pillow