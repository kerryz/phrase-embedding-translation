"""
Pseudo code

Initialization:
    Initialize W1 and get token2id dictionary through gensim lda
    Initialize neural network with W1, W2 = ID matrix

Goal: to learn W1 and W2
    1) Choose one source sentence and its corresponding N-best list (preferably randomly)
    Estimate gradients for W1 and W2 with this training sample
        ... procedure

"""

from __future__ import division

import numpy as np
import random, sys
from modules.cptm_neural_network import CPTMNeuralNetwork, get_error_term_dict
from modules.get_lists_and_dictionaries import get_everything
from modules.xbleu import xbleu, get_Ej_translation_probability_list
from gensim import corpora

debug_mode = True
debug_mode_verbose = False

learning_rate = 0.001  # for the neural network
smoothing_factor = 10  # see Equation 6 in the paper


def main(source_file_name, n_best_list_file_name, sbleu_score_list_file_name,
         learning_rate, smoothing_factor):
    print "Loading and initializing neural network"
    W1 = np.loadtxt("data/weight_initialization.gz")
    W2 = np.identity(100)
    nn = CPTMNeuralNetwork([W1.shape[0], 100, 100], [W1, W2])
    dictionary = corpora.Dictionary.load("data/dictionary.dict")

    training_set_size = 0
    # Each line in source_file is a source sentence.
    # source_file should end with an empty line
    with open(source_file_name, 'r') as source_file:
        for _ in source_file:
            training_set_size += 1
    training_set_size -= 1  # ends with empty line

    # uncomment to manually set training_set_size for testing purposes
    #training_set_size = 30
    training_order_list = range(training_set_size)
    # uncomment to randomize training samples
    # (should be done for deployment, however might be good with reproducable results while testing)
    #random.shuffle(training_order_list)

    test_set_size = max(10, int(0.1*training_set_size))
    # uncomment to manually set test_set_size for testing purposes
    #test_set_size = 5
    test_order_list = training_order_list[-test_set_size:]
    training_order_list = training_order_list[:-test_set_size]
    print "Training sample size:", training_set_size, "(nr of source sentences)"
    print "Test sample size:", test_set_size, "(nr of source sentences)"

    # initialize variables
    d_theta_old = [0, 0]  # momentum terms

    # calculate average loss function value of test samples
    print "Calculating average loss function value of test samples using initial weights"
    initial_loss_value_test_set = get_average_loss_value_of_test_sample(
        test_order_list, nn, dictionary, source_file_name,
        n_best_list_file_name, sbleu_score_list_file_name,
        smoothing_factor)
    print "Average loss function value:", initial_loss_value_test_set
    loss_value_history = [initial_loss_value_test_set]
    converged = False
    epoch_count = 0

    # For debug
    xBleu_history = []
    xBleu_change_history = []

    seen_nan = 0

    # train until overfit (early stop)
    print
    print "Start training..."
    while not converged:
        theta_previous = nn.weights
        for list_index, i in enumerate(training_order_list):
            (phrase_pair_dict_all, phrase_pair_dict_n_list,
                total_base_score_list, sbleu_score_list) \
                = get_everything(i, source_file_name,
                                 n_best_list_file_name, sbleu_score_list_file_name)
            xblue_i, Ej_translation_probability_list = xbleu(
                nn, total_base_score_list, sbleu_score_list,
                phrase_pair_dict_n_list, dictionary, smoothing_factor)

            error_term_dict_i = get_error_term_dict(
                phrase_pair_dict_all, phrase_pair_dict_n_list,
                sbleu_score_list, xblue_i,
                Ej_translation_probability_list)

            d_theta_old = nn.update_mini_batch(
                phrase_pair_dict_all, learning_rate, dictionary, error_term_dict_i, d_theta_old)

            if debug_mode:
                debug_print_weights_after_update(nn, d_theta_old, error_term_dict_i)

            # check if xBleu increases after each iteration for testing purposes
            # change to >> if True: << if you want to make this check and see output
            if False:
                xblue_i_after, _ = xbleu(nn, total_base_score_list, sbleu_score_list,
                                         phrase_pair_dict_n_list, dictionary, smoothing_factor)

                if np.isnan(xblue_i_after):
                    converged = True

                xBleu_history.append((i, xblue_i))
                xBleu_change_history.append(xblue_i_after - xblue_i)
                print "-------------------------------------------------------------"
                print "xBleu history: [(xBleu_before_gradient_descent, xBleu_after_gradient_descent)]"
                print xBleu_history
                print
                print "xBleu_change_history [xBleu_after - xBleu_before]"
                print xBleu_change_history
                print "-------------------------------------------------------------"

            print "Finished epoch nr", epoch_count, "training sample nr", list_index + 1,\
                  "(of %d)" % (training_set_size - test_set_size), "| source sentence nr", i+1

        epoch_count += 1
        print "====================================="
        print "Finished epoch number:", epoch_count
        print "====================================="

        # calculate loss function on test set after each epoch using updated weights
        print "Calculating loss function value on test set (%d samples) using new weights" % test_set_size
        loss_value_test_set = get_average_loss_value_of_test_sample(
            test_order_list, nn, dictionary,
            source_file_name, n_best_list_file_name, sbleu_score_list_file_name,
            smoothing_factor)
        loss_value_history.append(loss_value_test_set)
        print_loss_value_history(loss_value_history, test_set_size)

        # TODO: CONVERGENCE TEST, a simple approach is to stop once the loss function value
        # of this iteration is worse than the previous one
        #if loss_value_history[-2] > loss_value_history[-1]:
        if False:  # as of now, run forever. Change this once you have determined a good convergence criteria
            converged = True
            print "CONVERGED!!!!!!!!!!!!"
            print "Saving weights from previous epoch to file"
            np.savetxt('W1.gz', theta_previous[0])
            np.savetxt('W2.gz', theta_previous[1])
        else:
            print "No overfitting, keep training..."
            # uncomment to randomize training samples. 
            # commenting out this randomizer to get reproducable results during testing phase of development
            #random.shuffle(training_order_list)


def get_average_loss_value_of_test_sample(test_order_list, nn, dictionary,
                                          source_file_name, n_best_list_file_name,
                                          sbleu_score_list_file_name, smoothing_factor):
    loss_value_test_set = 0
    for t_i in test_order_list:
        (phrase_pair_dict_n_listase_pair_dict_all, phrase_pair_dict_n_list,
            total_base_score_list, sbleu_score_list) \
            = get_everything(t_i, source_file_name, n_best_list_file_name,
                             sbleu_score_list_file_name)
        xBlue_t_i, _ = xbleu(nn, total_base_score_list, sbleu_score_list,
                             phrase_pair_dict_n_list, dictionary, smoothing_factor)
        sys.stdout.write(str(-xBlue_t_i) + ", ")
        sys.stdout.flush()
        loss_value_test_set -= xBlue_t_i
    print
    return loss_value_test_set/len(test_order_list) if len(test_order_list) > 0 else float('nan')


def debug_print_weights_after_update(nn, d_theta, error_term_dict):
    size1, size2 = (d_theta[0].size, d_theta[1].size)
    sum_error_terms = 0
    for i, error in error_term_dict.iteritems():
        sum_error_terms += error
    avg_error_term = sum_error_terms / len(error_term_dict)
    print "     Average error term:", avg_error_term
    print "     Average weight change             | d_W1: %.9f, d_W2: %.9f" \
        % (sum(sum(d_theta[0]))/size1, sum(sum(d_theta[1]))/size2)
    print "     Average absolute weight change    | d_W1: %.9f, d_W2: %.9f" \
        % (sum(sum(np.absolute(d_theta[0])))/size1, sum(sum(np.absolute(d_theta[1])))/size2)


def print_loss_value_history(loss_value_history, test_set_size):
    print "Test set size:", test_set_size, "(nr of source sentences)"
    print "Old average loss function value (-xBleu) of test set, from previous epoch:"
    print loss_value_history[-2]
    print "Average loss function value using updated weights after gradient descent:"
    print loss_value_history[-1]
    print "Difference (new_loss_value - old_loss_value):"
    print loss_value_history[-1] - loss_value_history[-2]

    print
    print "***********************************************************"
    print "Loss function value after each epoch, starting from epoch 0"
    print loss_value_history
    print
    print "Loss function value change after each epoch, starting from epoch 1"
    print map(lambda (b, a): a - b, zip(loss_value_history[:-1], loss_value_history[1:]))
    print "************************************************************"
    print


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        learning_rate = sys.argv[3]
    if len(sys.argv) >= 5:
        smoothing_factor = sys.argv[4]
    main(sys.argv[1], sys.argv[2], "data/sbleu.txt", learning_rate, smoothing_factor)
