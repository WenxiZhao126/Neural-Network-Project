# Neural-Network-Project

# UVA MSDS 2018 Spring 2019

# Fang You (fy6vj)，Wenxi Zhao (wz8nx), Ruoyan Chen (rc3my), Shaoran Li (sl4bz)

# Overview:

This project aims to predict price movement of a designated stock during a fixed time frame using neural network models. The motivation comes from the increasing usage of electronic-trading platform in day-to-day trading activities, whereas the automation of movement of mid-price and price spread crossing becomes an essential part of every-day trading mechanism. By characterising the existing features in given dataset and creating new statistical features such as moving averages, we applied feed forward neural network (FFNN), convolutional neural network (CNN) and recurrent neural network (RNN) on training dataset and tested on validation dataset. After a comparison and discussion of the accuracies and losses of all three models, we reach to a conclusion that FFNN model works the best, with a training accuracy at 0.531 and test accuracy at 0.5, as well as a training loss level at 1.038 and test loss level at 1.04.

# Content:

./Output_CSV: after doing preprocessing and spliting train and test set, write out original data as csv files 
