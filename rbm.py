"""
Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import timeit
import Image
import numpy
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
import numpy.random as numpy_rng
import scipy.signal
convolve = scipy.signal.convolve


class RBM(object):
    def __init__(self, input=None, n_visible=1024, n_hidden=500, W=None,
        hbias=None, vbias=None, scale=0.01):

        if W is None:
            # initial_W = numpy.asarray(scale*numpy_rng.randn(n_visible, n_hidden),dtype=theano.config.floatX)
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            # initial_hbias = numpy.asarray(2*scale*numpy_rng.randn(n_hidden), dtype=theano.config.floatX)
            initial_hbias = numpy.zeros(n_hidden, dtype=theano.config.floatX)
            hbias = theano.shared(value=initial_hbias, name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            # initial_vbias = numpy.asarray(scale*numpy_rng.randn(n_visible), dtype=theano.config.floatX)
            initial_vbias = numpy.zeros(n_visible, dtype=theano.config.floatX)
            vbias = theano.shared(value=initial_vbias, name='vbias', borrow=True)

        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        # computing free energy
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def hid_expect(self, vis):
        # updating the hidden states
        activation = T.dot(vis, self.W) + self.hbias
        return [activation, T.nnet.sigmoid(activation)]

    def sample_h_v(self, v0_sample):
        # compute the activation of the hidden units given visibles samples
        pre_sigmoid_h1, h1_mean = self.hid_expect(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def vis_expect(self, hid):
        # updating the visible states
        activation = T.dot(hid, self.W.T) + self.vbias
        return [activation, T.nnet.sigmoid(activation)]

    def sample_v_h(self, h0_sample):
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.vis_expect(h0_sample)
        # get a sample of the visible given their activation
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def cost_updates(self, lr=0.1, persistent=None, k=1):

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_v(self.input)
        chain_start = ph_sample
        ([pre_sigmoid_nvs, nv_means, nv_samples, 
          pre_sigmoid_nhs, nh_means, nh_samples], updates) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that chain_start
            # is the initial state corresponding to the 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k)

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )
        return cross_entropy


# class CRBM(RBM):
#     def __init__(self, input=None, filter_shape, num_filters, pool_shape, 
#         binary=True, numpy_rng=None, theano_rng=None, scale=0.001):


#         # self.num_filters = num_filters
#         # self.weights = scale * rng.randn(num_filters, *filter_shape)
#         # self.vis_bias = scale * rng.randn()
#         # self.hid_bias = 2 * scale * rng.randn(num_filters)

#         # self._visible = binary and sigmoid or identity
#         self._pool_shape = pool_shape


#     ##########
#         self.num_filters = num_filters
#         self.n_visible = n_visible
#         self.n_hidden = n_hidden

#         if numpy_rng is None:
#             numpy_rng = numpy.random

#         if theano_rng is None:
#             theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

#         if W is None:
#             initial_W = scale * numpy_rng.randn(num_filters, *filter_shape)
#             W = theano.shared(value=initial_W, name='W', borrow=True)

#         if hbias is None:
#             # create shared variable for hidden units bias
#             initial_hbias = 2*scale*numpy_rng.randn(num_filters)
#             hbias = theano.shared(value=initial_hbias, name='hbias', borrow=True)

#         if vbias is None:
#             # create shared variable for visible units bias
#             initial_vbias = scale*numpy_rng.randn()
#             vbias = theano.shared(value=initial_vbias, name='vbias', borrow=True)

#         self.input = input
#         if not input:
#             self.input = T.matrix('input')
#         self.W = W
#         self.hbias = hbias
#         self.vbias = vbias
#         self.theano_rng = theano_rng
#         self.params = [self.W, self.hbias, self.vbias]


