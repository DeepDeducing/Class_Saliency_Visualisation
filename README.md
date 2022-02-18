# Deep Deducing on class saliency visualization and providing explainibility in deep neural network

This repository contains codes of deep deducing visualizing class saliency by optimizing input image by using gradient descent provided by error backpropagation.
To reproduce the results in the paper, simply run Deducing.py

The MNIST database is available at http://yann.lecun.com/exdb/mnist/

## Requirements

Packages requirements:

```
numpy
```


```
scipy
```

## Learning phase
To train sets of weight matrices in the paper, run this command:

```
Learning_xxx.py            
```


## Deducing phase
To use sets of trained weight matrices to start deducing, run this command:

```
Deducing.py              
```
