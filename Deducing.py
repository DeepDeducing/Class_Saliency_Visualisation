import os
import struct
import numpy as np
import gym
import math
from matplotlib import pyplot as plt

#--------------------------------------------------------------------

def load_mnist(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
    return images, labels

X_train, Y_train = load_mnist('mnist', kind = 'train')

#--------------------------------------------------------------------

def inverse_sigmoid(input):
    return np.log((input+ 0.0000000001) /(1-input + 0.0000000001))

def vectorizing(array_size, init, interv, input):
    array = np.zeros(array_size)
    array[int(array_size//2 - 1 + (input - init) // interv)] = 1
    return array

def quantifying(array_size, init, interval, input):
    array = np.zeros(array_size)
    if int( (input - init) // interval + 1) >= 0:
        array[ : int( (input - init) // interval + 1)] = 1
    return array

#-------------------------------------------------------------------




for trials in range(1):                                                  # <<<<<<<<<<<<




    from Brain_for_deducing import *                                     # <<<<<<<<<<<<
    network_size           = np.array([28 * 28, 100, 100, 10])           # <<<<<<<<<<<<
    beta                   = 0.1                                         # <<<<<<<<<<<<
    epoch_of_deducing      = 100000                                      # <<<<<<<<<<<<
    drop_rate              = 0.5                                         # <<<<<<<<<<<<
    Machine                = Brain(network_size, beta, epoch_of_deducing, drop_rate)

    weight_lists = list()
    slope_lists  = list()
    n_sets = 25                                                          # <<<<<<<<<<<<
    for n in range(n_sets):
        weight_name        = "100x100_25_0.000001_1m_0.2_[" + str(0 + n + 1) +  "]_weight_list.npy"   # <<<<<<<<<<<<
        slope_name         = "100x100_25_0.000001_1m_0.2_[" + str(0 + n + 1) +  "]_slope_list.npy"    # <<<<<<<<<<<<
        weight_list        = np.load(weight_name  , allow_pickle=True)
        slope_list         = np.load(slope_name   , allow_pickle=True)
        weight_lists.append(weight_list)
        slope_lists.append(slope_list)




    target_number                            = 0                                                    # <<<<<<<<<<<< The targeted class to extract saliency
    input_value                              = (np.random.random((1,28*28)) - 0.5) * 0.- 3.5        # <<<<<<<<<<<<
    input_resistor                           = np.ones_like(input_value)
    output_value                             = np.zeros((1,10))
    output_value[0,target_number]            = 1

    for i in range(epoch_of_deducing):
        random_index         = np.random.randint(np.array(weight_lists).shape[0])
        weight_list          = weight_lists[random_index]
        slope_list           = slope_lists[random_index]

        input_value          = Machine.deduce_batch(input_value,
                                                    input_resistor,
                                                    output_value,
                                                    weight_list, slope_list)




    plt.imshow(Machine.activator(input_value).reshape((28, 28)) , cmap='gray')
    plt.show()






