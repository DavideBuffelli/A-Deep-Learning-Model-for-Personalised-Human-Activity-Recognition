# Tests on DeepSense

The files in this folder perform the tests described in my [thesis](../DavideBuffelliThesis.pdf) (chapter 4) on the
original DeepSense framework and the customized framework.

## Files

* [test_main.py](test_main.py)
The code in this file simply launches the tests. 

* [test01.py](test01.py)
This test performs leave-one-out cross validation evaluating the performances of the [original DeepSense framework](../deepSense.py) when 
trained with the augmented dataset (130'000 examples) and with the non-augmented one (13'000 examples).

* [test02.py](test02.py)
This test performs leave-one-out cross validation evaluating the performances of the [customized DeepSense framework](../transferLearning/transferLearning.py).

* [test03.py](test03.py)
This test is aimed to demonstrate the effectiveness of the [customized DeepSense framework](../transferLearning/transferLearning.py) by training the base
network on the training set where the labels have been randomly permutated and then training and evaluating the final custom network on the same dataset
used in test02. 

## Requirements

* Python 3.x
* NumPy package
* TensorFlow 1.5.x or greater

## Acknowledgments

* [TensorFlow](https://www.tensorflow.org/) - TensorFlow official website.
* [NumPy](http://www.numpy.org) - NumPy official website.