# Pre-Processing for DeepSense

This is my implementation of the pre-processing of the Heterogeneity Activity Recognition Data Set (HHAR) dataset for the DeepSense framework.
A detailed explanation of the pre-processing can be found in the original DeepSense article (link below) and in my [thesis](../DavideBuffelliThesis.pdf) (chapter 3).

## Instructions

First of all it is necessary to download the HHAR dataset (you'll find a link in the Acknowledgments section). 
Before executing the file you need to edit the paths of the directories according to where they are saved in your system.
Then the files are meant to be executed in the following order:

* [sepUsers.py](sepUsers.py)
The code in this file creates one folder for each user, and inside each folder it creates two .csv files that will
contain accelerometer and gyroscope data for that user. 

* [mergeSensors.py](mergeSensors.py)
This will "merge" the measurements from accelerometer and gyroscope, doing all the necessary pre-processing.

* [prepareCV.py](prepareCV.py)
This file contains a script that organizes the data in folders for leave-one-out cross validation.

## Requirements

* Python 3.x
* NumPy package
* SciPy package

## Acknowledgments

* [HHAR Dataset](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) - Heterogeneity Activity Recognition Data Set.
* [DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing](https://arxiv.org/abs/1611.01942) - The paper for the DeepSense framework by Shuochao Yao, Shaohan Hu, Yiran Zhao, Aston Zhang, Tarek Abdelzaher.
* [HHAR-Data-Process](https://github.com/yscacaca/HHAR-Data-Process) - The code for the pre-processing created by the authors of DeepSense.
* [NumPy](http://www.numpy.org) - NumPy official website.
* [SciPy](https://www.scipy.org/) - SciPy official website.