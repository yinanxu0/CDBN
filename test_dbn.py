

import cPickle
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import load_CIFAR as LF
from DBN import DBN
from load_data import load_data

DEBUG_SET = 1

def test_DBN(finetune_lr=0.05, pretraining_epochs=20, pretrain_lr=0.01,
            k=1, training_epochs=1000, batch_size=10,DEBUG=0):

    # datasets = LF.load_cifar()
    # IMAGE_SIZE = 32
    datasets = load_data(DEBUG=DEBUG)
    IMAGE_SIZE = 28

    if DEBUG:
        N1 = 400
        N2 = 600
    else:
        N1 = 40000
        N2 = 50000
    x, y = datasets[0]
    train_set_x = x[:N1]
    train_set_y = y[:N1]
    valid_set_x = x[N1:N2]
    valid_set_y = y[N1:N2]
    test_set_x, test_set_y = datasets[1]
    datasets = [(train_set_x, train_set_y),
                (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0].eval()/batch_size
    n_valid_batches = valid_set_x.shape[0].eval()/batch_size
    n_test_batches = test_set_x.shape[0].eval()/batch_size

    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(n_ins=IMAGE_SIZE**2, hidden_layers_sizes=[500, 500],
              n_outs=10)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost %.4f' % (i+1, epoch+1, numpy.mean(c))

    end_time = timeit.default_timer()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_fn, test_fn = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many minibatches before
                                  # checking the network on the validation
                                  # set; in this case we check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [validate_fn(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %.2f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_fn(i) for i in xrange(n_test_batches)]
                    test_score = 1-numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %.2f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # save the best model
                    f = file('best_model_dbn.pkl', 'w')
                    cPickle.dump(dbn, f, protocol=cPickle.HIGHEST_PROTOCOL)
                    f.close()

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        ('Optimization complete with best validation score of %.2f %%, '
            'with test performance %.2f %%') 
        %((1-best_validation_loss)*100., test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

if __name__ == '__main__':
    test_DBN(DEBUG=DEBUG_SET)