from __future__ import division

import numpy as np

debug_mode = False
debug_mode_verbose = False

momentum_term = 0.9


class CPTMNeuralNetwork():

    def __init__(self, sizes, weights=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        if weights:
            self.weights = weights
        else:
            weights = []
            # w1 is dx100 matrix where each column sums to 1
            w1 = np.random.dirichlet(np.ones(sizes[0]), size=sizes[1]).transpose()
            w2 = np.identity(sizes[1])
            weights.append(w1)
            weights.append(w2)
            self.weights = np.array(weights)

    def get_z(self, weights, a):
        """
        z is the input vector to a layer.
        E.g. for layer 1 (the first hidden layer) z = W_1.tranpose * x
        where x is the input layer
        """
        return np.dot(weights.transpose(), a)

    def get_z_sparse(self, W1, word_id_count_dict):
        """
        Same as function get_z, using a word_id_count_dict instead.
        See function get_word_id_count_dict
        """
        z1 = np.zeros((self.sizes[1], 1))
        for i, cell in enumerate(z1):
            for word_index, count in word_id_count_dict.iteritems():
                # Use transposed indices instead of transposing whole W1
                cell += W1[word_index, i] * count
        return z1

    def get_y(self, z):
        return np.vectorize(np.tanh)(z)

    def sparse_array_to_d_W1(self, word_id_count_dict, not_sparse):
        """
        Optimizing Equation 10 in computer,
        where x_f and x_e are sparse vectors.

        @not_sparse : not_sparse is a 1x100 matrix
        """
        W1_gradient = np.zeros((self.sizes[0], self.sizes[1]))
        for row_index, count in word_id_count_dict.iteritems():
            for column_index, _ in enumerate(not_sparse[0]):  # not_sparse is 1x100
                W1_gradient[row_index, column_index] += not_sparse[0, column_index] * count
        return W1_gradient

    def get_theta_gradients(self, f, e, dictionary):
        """
        @x_f : bag of words of foreign phrase
        @x_e : bag of words of English phrase
        """
        W1 = self.weights[0]
        W2 = self.weights[1]

        f_word_id_count_dict = get_word_id_count_dict(f, dictionary)
        e_word_id_count_dict = get_word_id_count_dict(e, dictionary)

        # get input and output for each layer, given f as input
        z1_f = self.get_z_sparse(W1, f_word_id_count_dict)
        y1_f = self.get_y(z1_f)
        z2_f = self.get_z(W2, y1_f)
        y2_f = self.get_y(z2_f)

        # get input and output for each layer, given e as input
        z1_e = self.get_z_sparse(W1, e_word_id_count_dict)
        y1_e = self.get_y(z1_e)
        z2_e = self.get_z(W2, y1_e)
        y2_e = self.get_y(z2_e)

        # calculate W2_gradient
        W2_gradient = np.dot(y1_f, (y2_e * tanh_prime(z2_f)).transpose())\
            + np.dot(y1_e, (y2_f * tanh_prime(z2_e)).transpose())

        # calculate W1_gradient
        tmp1 = (np.dot(W2, y2_e * tanh_prime(z2_f)) * tanh_prime(z1_f)).transpose()
        tmp2 = (np.dot(W2, y2_f * tanh_prime(z2_e)) * tanh_prime(z1_e)).transpose()
        # W1_gradient = np.dot(x_f, tmp1 + np.dot(x_e, tmp2)
        W1_gradient = np.zeros((self.sizes[0], self.sizes[1]))
        W1_gradient += self.sparse_array_to_d_W1(f_word_id_count_dict, tmp1)
        W1_gradient += self.sparse_array_to_d_W1(e_word_id_count_dict, tmp2)

        return W1_gradient, W2_gradient

    def feedforward(self, x):
        """
        Returns the output vector of the network, if @x is the input
        """
        for w in self.weights:
            x = np.vectorize(np.tanh)(np.dot(w.transpose(), x))
        return x

    def feedforward_sparse_bow(self, phrase, dictionary):
        # [(word_index, count)]
        word_count_dict = {}
        for word in phrase.split(" "):
            word = unicode(word, 'utf-8')
            if word in dictionary.token2id:
                word_index = dictionary.token2id[word]
                word_count_dict[word_index] = word_count_dict.get(word_index, 0) + 1

        W1 = self.weights[0]
        W2 = self.weights[1]
        z1 = np.zeros((self.sizes[1], 1))
        for i, cell in enumerate(z1):
            for word_index, count in word_count_dict.iteritems():
                # Use transposed indices instead of transposing whole W1
                cell += W1[word_index, i] * count
        y1 = self.get_y(z1)
        z2 = self.get_z(W2, y1)
        y2 = self.get_y(z2)

        return y2

    def update_mini_batch(self, mini_batch, eta, dictionary, error_term_dict, d_theta_old):
        """
        Update the network's weights (W_1, W_2).
        Calculates the gradients (Equation 7 in the paper)
            sum_(f, e) {dL/dtheta = error_term * theta gradients}

        @mini_batch : dictionary of all phrase pairs. mini_batch[(f, e)] = c
                            where f = source phrase, e = target phrase, c = count (# times observed)
                            e.g. dictionary[("vamos", "let's go")] = 3
                      observed in a source sentence F_i and its corresponding
                      N-best list GEN(F_i)
        @eta : float, learnig factor
        @dictionary : class gensim.corpora.Dictionary
        @error_term_dict : dict in form of: error_term_dict[(f, e)] = error_term(f, e)
        """
        # Gradients for W1 and W2
        d_W1 = np.zeros((self.sizes[0], self.sizes[1]))
        d_W2 = np.zeros((self.sizes[1], self.sizes[2]))

        # Equation 7 in the paper.
        # For each sentence pair (f, e) observed between source sentence F_i
        # and its corresponding N-best list GEN(F_i)
        #   calculate error_f_e and the gradient it contributes to
        for f, e in mini_batch.keys():

            W1_gradient, W2_gradient = self.get_theta_gradients(f, e, dictionary)
            count = mini_batch[(f, e)]
            error_term = error_term_dict[(f, e)]

            if debug_mode_verbose:
                print "For phrase pair (f, e):", f, e
                print "error term:", error_term
                print "count:", count

            d_W1 += (-1)*count * error_term * W1_gradient
            d_W2 += (-1)*count * error_term * W2_gradient

        if debug_mode or debug_mode_verbose:
            print "--------------------------------------------"
            print "Average absolute change for an element in W1"
            print sum(sum(np.absolute(d_W1))) / d_W1.size
            print
            print "Average absolute change for an element in W2"
            print sum(sum(np.absolute(d_W2))) / d_W2.size
            print "--------------------------------------------"

        # add momentum term
        d_W1 += momentum_term * d_theta_old[0]
        d_W2 += momentum_term * d_theta_old[1]

        # gradient descend
        self.weights[0] -= eta * d_W1
        self.weights[1] -= eta * d_W2

        return [d_W1, d_W2]


def tanh_prime(x):
    return np.vectorize(lambda x: 1 - (np.tan(x)**2))(x)


def phrase_to_bow_vector(phrase, dictionary):
    """
    Returns the bag of word vector representation of a phrase.
    @phrase : string, where each word is separated by a white space
    @dictionary : gensim.corpora.Dictionary
    """
    x = np.zeros((len(dictionary), 1))
    for word in phrase.split(" "):
        word = unicode(word, 'utf-8')
        if word in dictionary.token2id:
            word_index = dictionary.token2id[word]
            x[word_index] += 1
    return x


def get_error_term_dict(
        phrase_pair_dict_all, phrase_pair_list_hypothesis,
        sBleu_list, xBleu,
        Ej_translation_probability_list,
        new_feature_weight=1):
    """
    From Equation 14
    Returns the error term for each phrase pair observed
    between F_i and all sentences in GEN(F_i)

    @phrase_pair_dict_all : dictionary of all phrase pairs. mini_batch[(f, e)] = c
                where f = source phrase, e = target phrase, c = count (# times observed)
                e.g. dictionary[("vamos", "let's go")] = 3
                observed in a source sentence F_i and all sentences in its corresponding
                N-best list GEN(F_i)
    @sBleu_list : list of sBleu values for all sentences in GEN(F_i)
    @xBleu
    @Ej_translation_probability_list :
        list of translation probabilitiesfor all sentences in GEN(F_i)
    @phrase_pair_list_hypothesis : list of dicts, where each dict
        is the phrase pair dict between F_i and a hypothesis translation in GEN(F_i)
    """
    # See equation 14
    U_list = map(lambda sBleu: sBleu - xBleu, sBleu_list)

    error_dict = {}
    for f, e in phrase_pair_dict_all.keys():
        for j, phrase_pair_dict_j in enumerate(phrase_pair_list_hypothesis):
            U_j = U_list[j]
            P_j = Ej_translation_probability_list[j]
            if (f, e) in phrase_pair_dict_j:
                N_j = phrase_pair_dict_j[(f, e)]
            else:
                N_j = 0
            error_dict[(f, e)] = U_j * P_j * new_feature_weight * N_j
    return error_dict


def new_feature_value(nn, phrase_pair_dict, dictionary):
    """
    Calculates the new feature value of a translation,
    using Equation 4 in the paper.
    Returns its logarithm.

    @nn : class CPTMNeuralNetwork, our neural network
    @phrase_pair_dict : dictionary of all phrase pairs. phrase_pair_dict[(f, e)] = c
                            where f = source phrase, e = target phrase, c = count (# times observed)
                            e.g. dictionary[("vamos", "let's go")] = 3
                        observed between a source sentence F_i
                        and the translation whose feature value we want to calculate
    @dictionary : class gensim.corpora.Dictionary
    """
    feature_value = 0
    for f, e in phrase_pair_dict.keys():

        #x_f = phrase_to_bow_vector(f, dictionary)
        #x_e = phrase_to_bow_vector(e, dictionary)

        #y_f = nn.feedforward(x_f)
        #y_e = nn.feedforward(x_e)

        y_f = nn.feedforward_sparse_bow(f, dictionary)
        y_e = nn.feedforward_sparse_bow(e, dictionary)

        count = phrase_pair_dict[(f, e)]
        feature_value += count * np.dot(y_f.transpose(), y_e)
    return feature_value[0, 0]


def get_word_id_count_dict(phrase, dictionary):
    """
    Returns a dict of form:
    {word_id : word_count}

    Where word_id maps to the number of times it appears in @phrase
    """
    word_count_dict = {}
    for word in phrase.split(" "):
        word = unicode(word, 'utf-8')
        if word in dictionary.token2id:
            word_index = dictionary.token2id[word]
            word_count_dict[word_index] = word_count_dict.get(word_index, 0) + 1
    return word_count_dict
