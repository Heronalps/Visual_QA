#!/usr/bin/env bash

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

sudo apt-get install default-jdk

jar xf train2014.zip
jar xf val2014.zip
jar xf test2015.zip

rm -rf train2014.zip val2014.zip test2015.zip

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc

p

sudo apt-get install python-dev python-pip libcupti-dev
sudo pip install --upgrade tensorflow-gpu==1.4.0



pip3 install tensorflow-gpu

pip3 install pandas

pip3 install tqdm

pip3 install nltk

pip3 install opencv-python

pip3 install scipy

pip3 install Pillow