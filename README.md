# A Deep Learning Model for Personalised Human Activity Recognition

This repository contains the code I developed for my Master's Degree in Computer Engineering thesis. I graduated with honours at the University of Padova, where I was supervised by professor Fabio Vandin. 

You can find my thesis at the following link: [http://tesi.cab.unipd.it/62146/](http://tesi.cab.unipd.it/62146/)

__Please refer at the following repository: <https://github.com/DavideBuffelli/TrASenD> for a more updated version, and a link to a paper version of this work.__

## Cite

If you use the code in this repository, please cite the following work:

```A Deep Learning Model for Personalized Human Activity Recognition, Buffelli D., University of Padova, 2019```

## File Organization

* [deepSense.py](deepSense.py)
My implementation of the DeepSense model.

* [pre-processing](pre-processing)
This folder contains the files that pre-process the data from the HHAR dataset.

* [transferLearning](transferLearning)
This folder contains the custom DeepSense model I developed. This model adapts to a specific user, improving the accuracy of the predictions.

* [tests](tests)
The folder contains the files implementing the tests that have been done.

## Requirements

* Python 3.x
* NumPy package
* TensorFlow 1.5.x (haven't tested it with other versions)

## Related links

* [HHAR Dataset](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) - Heterogeneity Activity Recognition Data Set.
* [DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing](https://arxiv.org/abs/1611.01942) - The DeepSense original paper by Shuochao Yao, Shaohan Hu, Yiran Zhao, Aston Zhang, Tarek Abdelzaher.
* [DeepSense](https://github.com/yscacaca/DeepSense) - The code created by the authors of the paper.
* [HHAR-Data-Process](https://github.com/yscacaca/HHAR-Data-Process) - The code for the pre-processing created by the authors of DeepSense.
* [TensorFlow](https://www.tensorflow.org/) - TensorFlow official website.
* [NumPy](http://www.numpy.org) - NumPy official website.

## Licence
Refer to the the file [LICENCE](LICENCE).
