#############################################################
#            Created by Yinan Xu, 12/03/2015                #
#            Copyright @ Yinan Xu                           #
#############################################################

import cPickle
import timeit
import numpy
import theano
import theano.tensor as T
import load_CIFAR as LF
from logistic_sgd import LogisticRegression


def sgd_optimization(learning_rate=0.1, n_epochs=1000, batch_size=200):

    datasets = LF.load_cifar()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

    print '... building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=1024, n_out=10)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    cost = classifier.neg_log_hood(y)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'

    test_score = 0.
    start_time = timeit.default_timer()
    prev_avg_cost = 0
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_cost_vector=[train_model(i) for i in xrange(n_train_batches)]
        train_cost = numpy.mean(train_cost_vector)/batch_size
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        test_score = numpy.mean(test_losses)
        print 'epoch %d with train cost %.2f, test error %.2f%%'% (epoch,train_cost,test_score*100.)

        if train_cost<1920:

            test_losses_opt = [test_model(i) for i in xrange(n_test_batches)]
            test_score_opt = numpy.mean(test_losses_opt)
            # save the best model
            f = file('best_model_logistic.pkl', 'w')
            cPickle.dump(classifier, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            break
        prev_avg_cost=train_cost

    end_time = timeit.default_timer()
    print 'Optimization complete with test performance %.2f %%' %(test_score_opt*100.)
    print 'The code runs for %.1fs' % (end_time - start_time)


def predict():

    classifier = cPickle.load(file('best_model_logistic.pkl','r'))
    datasets = LF.load_cifar()
    test_set_x, test_set_y = datasets[1]
    test_set_x = test_set_x.get_value()

    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.errors(test_set_y))

    test_score = predict_model(test_set_x)
    print "Predicted accuracy for the test set: %.2f%%" % (test_score*100.)

if __name__ == '__main__':
    # sgd_optimization()
    # predict()
