#############################################################
#            Created by Yinan Xu, 11/19/2015                #
#            Copyright @ Yinan Xu                           #
#############################################################

import cPickle
import gzip
import theano
import numpy
import theano.tensor as T

SIZE = 1024

def splitDataset(dataX, dataY):
    dataX = numpy.transpose(numpy.asarray(dataX, dtype=theano.config.floatX))
    dataX_gray = 0.21*dataX[0:SIZE]+0.72*dataX[SIZE:2*SIZE]+0.07*dataX[2*SIZE:3*SIZE]
    shared_x = theano.shared(numpy.transpose(dataX_gray), borrow=True)
    shared_y = theano.shared(numpy.asarray(dataY, dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, 'int32')

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar(SIZE_TRAIN = 5):

    print '... load data'
    path_name = '../cifar-10-batches-py/'

    # load train data
    batch_name_base = 'data_batch_'
    train_data = []
    train_label = []
    for i in range(SIZE_TRAIN):
        batch_name = path_name+batch_name_base+str(i+1)
        batch_data = unpickle(batch_name)
        train_data.extend(batch_data['data'])
        train_label.extend(batch_data['labels'])
    
    # load test data
    test_batch_name = path_name+'test_batch'
    test_batch = unpickle(test_batch_name)
    test_data = test_batch['data']
    test_label = test_batch['labels']

    # split data
    x_train, y_train = splitDataset(train_data, train_label)
    x_test, y_test = splitDataset(test_data, test_label)

    return [(x_train, y_train), (x_test, y_test)]
