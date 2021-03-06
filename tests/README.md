# Tests on DeepSense

The files in this folder perform the tests described in my [thesis](http://tesi.cab.unipd.it/62146/) (chapter 5) on the
original DeepSense framework and the customized framework.

## Files

* [test_main.py](test_main.py)
The code in this file simply launches the tests. 

* [test01.py](test01.py)
This test performs leave-one-out cross validation evaluating the performances of the [original DeepSense framework](../deepSense.py) when 
trained with the augmented dataset (circa 120'000 examples) and with the non-augmented one (circa 12'000 examples).

* [test02.py](test02.py)
This test performs leave-one-out cross validation evaluating the performances of the [customized DeepSense framework](../transferLearning/transferLearning.py).

* [test03.py](test03.py)
This test is aimed to demonstrate the effectiveness of the [customized DeepSense framework](../transferLearning/transferLearning.py) by training the base
network on the training set where the labels have been randomly permutated and then training and evaluating the final custom network on the same dataset
used in test02 (therefore on regular data). 

## Requirements

* Python 3.x
* NumPy package
* TensorFlow 1.5.x (haven't tested it with other versions)

## Related Links

* [TensorFlow](https://www.tensorflow.org/) - TensorFlow official website.
* [NumPy](http://www.numpy.org) - NumPy official website.
