import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


def shared_dataset(data_xy, borrow=True):

    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32')


def load_data(DEBUG=0):

    dataset = 'mnist.pkl.gz'
    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    if DEBUG:
        N1 = 800
        N2 = 200
        N3 = 400
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        test_set_x, test_set_y = test_set
        train_set = (train_set_x[:N1], train_set_y[:N1])
        valid_set = (valid_set_x[:N2], valid_set_y[:N2])
        test_set = (test_set_x[:N3], test_set_y[:N3])

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

