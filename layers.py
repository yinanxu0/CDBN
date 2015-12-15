#############################################################
#            Created by Yinan Xu, 12/03/2015                #
#            Copyright @ Yinan Xu                           #
#############################################################

import numpy
import theano
import theano.tensor as T
import numpy.random as numpy_rng

class LogisticLayer(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX ),
            name='W', borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX ),
            name='b', borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def neg_log_hood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                numpy_rng.uniform(
                    low = -4*numpy.sqrt(6. / (n_in + n_out)),
                    high = 4*numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b = theano.shared(
                value=numpy.zeros((n_out,),dtype=theano.config.floatX), 
                name='b', borrow=True
                )

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = T.nnet.sigmoid(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]

