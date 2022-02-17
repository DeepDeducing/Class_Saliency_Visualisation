import os
import struct
import numpy as np
import gym
import math

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

#--------------------------------------------------------------------

start_set     = 1   # <<<<<<<<<<<<
end_set       = 25  # <<<<<<<<<<<<

n_sets        = end_set - start_set + 1

for n in range(n_sets):




    from Brain_for_learning import *
    network_size              = np.array([28 * 28, 100, 100, 10])                 # <<<<<<<<<<<<
    slope                     = 25                                                # <<<<<<<<<<<<
    alpha                     = 0.000001                                          # <<<<<<<<<<<<
    epoch_of_learning         = 1000000                                           # <<<<<<<<<<<<
    drop_rate                 = 0.2                                               # <<<<<<<<<<<<
    momentum_rate             = 0.9                                               # <<<<<<<<<<<<

    Machine                   = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)

    retrain = False                                     # <<<<<<<<<<<<
    if retrain == True:
        Machine.weight_list            = np.load("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_weight_list.npy"          , allow_pickle=True)
        Machine.slope_list             = np.load("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_slope_list.npy"           , allow_pickle=True)
        Machine.weight_list_momentum   = np.load("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_weight_list_momentum.npy" , allow_pickle=True)
        Machine.slope_list_momentum    = np.load("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_slope_list_momentum.npy"  , allow_pickle=True)




    for i_episode in range(epoch_of_learning):

        print(i_episode)




        random_index = np.random.randint(X_train.shape[0])


        X = X_train[random_index]
        Y = Y_train[random_index]
        Y = np.eye(10)[Y]
        Machine.learn_batch(  np.atleast_2d(X/255),
                              np.atleast_2d(Y) )




    np.save("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_weight_list"             , Machine.weight_list                 ) # <<<<<<<<<<<<
    np.save("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_slope_list"              , Machine.slope_list                  ) # <<<<<<<<<<<<
    np.save("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_weight_list_momentum"    , Machine.weight_list_momentum        ) # <<<<<<<<<<<<
    np.save("100x100_25_0.000001_1m_0.2_[" + str(start_set + n) +  "]_slope_list_momentum"     , Machine.slope_list_momentum         ) # <<<<<<<<<<<<





