import numpy as np
from cptm_neural_network import CPTMNeuralNetwork


def test_get_z_sparse():
    """
    function CPTMNeuralNetwork.get_z_sparse
    implements an optimized algorithm for calculating
    the input z1 to the hidden layer. See section 5.1.1 in the paper
    z1 = numpy.dot(W1.transpose(), x)
    Test if optization works
    """
    d1 = 99
    d2 = 10

    W1 = np.random.randn(d1, d2)
    W2 = np.identity(d2)
    nn = CPTMNeuralNetwork(sizes=[d1, d2, d2], weights=[W1, W2])
    x = np.random.randint(5, size=(d1, 1))  # x is d1x1

    word_id_count_dict = {}
    for word_id, count in enumerate(x):
        word_id_count_dict[word_id] = word_id_count_dict.get(word_id, 0) + count

    r1 = np.dot(W1.transpose(), x)
    r2 = nn.get_z_sparse(W1, word_id_count_dict)

    test_passed = np.array_equal(r1, r2)
    print "Test result:", test_passed, "for test_get_z_sparse"
    return test_passed


def test_sparse_array_to_d_W1():
    """
    function CPTMNeuralNetwork.sparse_array_to_d_W1
    implements an optimized algorithm for Equation 10 in paper.
    Test if optization works
    """
    d1 = 99
    d2 = 10

    nn = CPTMNeuralNetwork(sizes=[d1, d2, d2])
    x = np.random.randint(5, size=(d1, 1))  # x is d1x1
    not_sparse = np.random.randn(1, d2)  # not_sparse is 1xd2 array

    word_id_count_dict = {}
    for word_id, count in enumerate(x):
        word_id_count_dict[word_id] = word_id_count_dict.get(word_id, 0) + count

    r1 = np.dot(x, not_sparse)
    r2 = nn.sparse_array_to_d_W1(word_id_count_dict, not_sparse)

    test_passed = np.array_equal(r1, r2)
    print "Test result:", test_passed, "for test_sparse_array_to_d_W1"
    return test_passed


if __name__ == "__main__":
    test_get_z_sparse()
    test_sparse_array_to_d_W1()
