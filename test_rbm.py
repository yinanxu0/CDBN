#############################################################
#            Created by Yinan Xu, 11/19/2015                #
#            Copyright @ Yinan Xu                           #
#############################################################

import timeit
import Image
import numpy
import os
import theano
import theano.tensor as T
from utils import tile_raster_images
import load_CIFAR as LF
from rbm import RBM

IMAGE_SIZE = 32
IMAGE_SIZE_PLUS = IMAGE_SIZE+1
DEBUG_SET = 0
from load_data import load_data

def test_rbm(learning_rate=0.1, training_epochs=10, batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):

    # datasets = load_data()
    # IMAGE_SIZE = 28
    # IMAGE_SIZE_PLUS = IMAGE_SIZE+1

    if DEBUG_SET:
        SIZE_TRAIN_SET=1
    else:
        SIZE_TRAIN_SET=5
    datasets = LF.load_cifar(SIZE_TRAIN=SIZE_TRAIN_SET,DEBUG=DEBUG_SET)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    n_train_batches = train_set_x.get_value().shape[0]/batch_size

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rbm = RBM(input=x, n_visible=IMAGE_SIZE*IMAGE_SIZE, n_hidden=n_hidden)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.cost_updates(lr=learning_rate, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    train_rbm = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = [train_rbm(i) for i in xrange(n_train_batches)]

        print 'Training epoch %d, cost is %.2f.' % (epoch+1, numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(IMAGE_SIZE, IMAGE_SIZE),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time/60.))

    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = numpy.random.randint(number_of_test_samples - n_chains)
    vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    ([_, _, _, _, vis_expects, vis_samples], updates) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [vis_expects[-1], vis_samples[-1]],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    # n_samples=10, n_chains=20
    image_data = numpy.zeros(
        (IMAGE_SIZE_PLUS*n_samples+1, IMAGE_SIZE_PLUS*n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[IMAGE_SIZE_PLUS*idx: IMAGE_SIZE_PLUS*idx+IMAGE_SIZE, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(IMAGE_SIZE, IMAGE_SIZE),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')

if __name__ == '__main__':
    test_rbm()
