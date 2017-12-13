# Utilities for the NN exercises in ISTA 421, Introduction to ML
from __future__ import print_function
import numpy as np
import math
import visualize
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    # Derivative Computation:
    # https://www.wolframalpha.com/input/?i=sigmoid
    # THIS DID NOT WORK!!! DONT USE THIS.
    # return np.exp(x) / (np.exp(x) + 1)**2


    # The Logistic Sigmoid Activation Function
    # https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
    # g'(x) = g(x) * [1 - g(x)]
    sig = sigmoid(x)
    
    return sig * (1-sig)

# -------------------------------------------------------------------------

def r(fan_in, fan_out):
    return np.sqrt(6/(fan_in + fan_out))

def initialize(hidden_size, visible_size):
    """
    Sample weights uniformly from the interval [-r, r] as described in lecture 23.
    Return 1d array theta (in format as described in Exercise 2)
    :param hidden_size: number of hidden units
    :param visible_size: number of visible units (of input and output layers of autoencoder)
    :return: theta array
    """

    # Uniform Initialization Scheme from Xavier Glorot (2010a)

    ### YOUR CODE HERE ###
    w1 = np.random.uniform(low = -r(visible_size, hidden_size),
                           high = r(visible_size, hidden_size),
                           size = visible_size * hidden_size)
    w2 = np.random.uniform(low = -r(hidden_size, visible_size),
                           high = r(hidden_size, visible_size),
                           size = hidden_size * visible_size)
    b1 = np.zeros((hidden_size))
    b2 = np.zeros((visible_size))


    theta = np.concatenate((w1,w2,b1,b2))

    return theta




# -------------------------------------------------------------------------

def autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, data):
    """
    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    m   = data.shape[1]
    len = visible_size * hidden_size
    
    w1 = theta[0 : len].reshape((hidden_size, visible_size))             # (h,v)
    w2 = theta[len : 2*len].reshape((visible_size, hidden_size))         # (v,h)
    b1 = theta[2*len : 2*len + hidden_size].flatten()                    # (h)
    b2 = theta[2*len + hidden_size: ].flatten()                          # (v)
    
    
    
    # FORWARD PROPAGATION (Lecture 24, Slides 11-13)
    # HW5 #4: Vectorized Implementation of Forward Propagation
    # Code moved to autoencoder_feedforward(theta, visible_size, hidden_size, data)

    tau = autoencoder_feedforward(theta, visible_size, hidden_size, data)
    z2  = tau[0 : hidden_size]
    a2  = tau[hidden_size : 2*hidden_size]
    z3  = tau[2*hidden_size : 2*hidden_size + visible_size]
    h   = tau[2* hidden_size + visible_size:]
    
    
    # COST FUNCTION (Equation on Lecture 24, Slide 15)
    #
    # J(W,b) = Squared Error Term + Weight Decay Term (for regularization)
    #
    #        = Sum{m=1...m} ||h_{W,b}(x^(i)) - y^(i)||^2 / (2m) + lambda/2 *  \
    #          Sum{l=1...n_l-1} Sum{i=1...s_l} Sum{j=1...s_l+1} (W_{j,i}^(i))^2
    #
    #   where
    #       m = # training pairs {(x^(1), y^(1)), ... , (x^(m), y^(,))}

    cost = np.linalg.norm(h - data)**2/2./m + lambda_/2. * (np.linalg.norm(w1)**2 + np.linalg.norm(w2)**2)
    
    
    
    # BACKPROPAGATION (Lecture 24, Slides 15-16)
    # Step 1: Perform feedforward pass, computing activations for layers L_{2...n}.
    #         Completed above.
    
    # Step 2: Compute "error terms." (Slide 16)
    #         delta_i^(l) = -[y_i - a_i^(l)] * f'[z_i^(l)]
    delta3 = -(data - h) * derivative(z3)                               # (v,m)
    delta2 = w2.T.dot(delta3) * derivative(z2)                          # (h,m)

    # Step 3: Compute partial derivatives. (Slide 15)
    #         partial J / partial W^(l) = a_j^(l) * delta_i^(l+1)
    #         partial J / partial b_i^(l) = delta_i^(l+1)

    w1_grad = delta2.dot(data.T) / m + lambda_ * w1
    w2_grad = delta3.dot(a2.T) / m + lambda_ * w2
    b1_grad = np.sum(delta2, axis=1) / m
    b2_grad = np.sum(delta3, axis=1) / m

    
    grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten(), b1_grad, b2_grad))

    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, data):
    """
    Version of cost and grad that incorporates the hidden layer sparsity constraint
        rho_ : the target sparsity limit for each hidden node activation
        beta_ : controls the weight of the sparsity penalty term relative
                to other loss components

    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    m   = data.shape[1]
    len = visible_size * hidden_size

    w1 = theta[0 : len].reshape((hidden_size, visible_size))             # (h,v)
    w2 = theta[len : 2*len].reshape((visible_size, hidden_size))         # (v,h)
    b1 = theta[2*len : 2*len + hidden_size].flatten()                    # (h)
    b2 = theta[2*len + hidden_size: ].flatten()                          # (v)
    
    
    # FORWARD PROPAGATION (Lecture 24, Slides 11-13)
    # HW5 #4: Vectorized Implementation of Forward Propagation
    # Code moved to autoencoder_feedforward(theta, visible_size, hidden_size, data)
    
    tau = autoencoder_feedforward(theta, visible_size, hidden_size, data)
    z2  = tau[0 : hidden_size]                                          # (h,m)
    a2  = tau[hidden_size : 2*hidden_size]                              # (h,m)
    z3  = tau[2*hidden_size : 2*hidden_size + visible_size]             # (v,m)
    h   = tau[2* hidden_size + visible_size:]                           # (v,m)
    
    
    
    # COST FUNCTION (Equation on Lecture 24, Slide 15)
    #
    # J(W,b) = Squared Error Term + Weight Decay Term (for regularization)
    #
    #        = Sum{m=1...m} ||h_{W,b}(x^(i)) - y^(i)||^2 / (2m) + lambda/2 *  \
    #          Sum{l=1...n_l-1} Sum{i=1...s_l} Sum{j=1...s_l+1} (W_{j,i}^(i))^2
    #
    #   where
    #       m = # training pairs {(x^(1), y^(1)), ... , (x^(m), y^(,))}
    
    cost = np.linalg.norm(h - data)**2/2./m + lambda_/2. * (np.linalg.norm(w1)**2 + np.linalg.norm(w2)**2)
    
    #---------------------------#
    
    # SPARSITY PARAMETER
    # http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
    #
    # rho-hat = Sum{i=1...m} [a_j^(2) * x^(i)] / m

    rhat = np.sum(a2, axis=1) / m

    # Kullback-Leibler (KL) Divergence
    # KL(rho || rho-hat) = rho * log(rho/rho_hat) + (1-rho) * log((1-rho)/(1-rho_hat))
    # Penalty = Sum{j=1...s_2} KL(rho || rho-hat)
    kl = np.sum(rho_ * np.log(rho_/rhat) + (1-rho_) * np.log((1-rho_)/(1-rhat)))
    cost += beta_ * kl
    
    #---------------------------#
    
    # BACKPROPAGATION (Lecture 24, Slides 15-16)
    # Step 1: Perform feedforward pass, computing activations for layers L_{2...n}.
    #         Completed above.

    # Step 2: Compute "error terms." (Slide 16)
    #         Use original equation for delta3.
    #         Use modified version for delta2.

    sparsity_der = beta_ * (-rho_/rhat + (1-rho_)/(1-rhat))
    
    delta3 = -(data - h) * derivative(z3)
    delta2 = (w2.T.dot(delta3) + np.repeat(sparsity_der,m).reshape((sparsity_der.shape[0],m))) * derivative(z2)


    #---------------------------#
    
    # Step 3: Compute partial derivatives. (Slide 15)
    #         partial J / partial W^(l) = a_j^(l) * delta_i^(l+1)
    #         partial J / partial b_i^(l) = delta_i^(l+1)
    w1_grad = delta2.dot(data.T) / m + lambda_ * w1
    w2_grad = delta3.dot(a2.T) / m + lambda_ * w2
    b1_grad = np.sum(delta2, axis=1) / m
    b2_grad = np.sum(delta3, axis=1) / m
    
    
    grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten(), b1_grad, b2_grad))

    #print("\tgrad shape:", grad.shape)

    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_feedforward(theta, visible_size, hidden_size, data):
    """
    Given a definition of an autoencoder (including the size of the hidden
    and visible layers and the theta parameters) and an input data matrix
    (each column is an image patch, with 1 or more columns), compute
    the feedforward activation for the output visible layer for each
    data column, and return an output activation matrix (same format
    as the data matrix: each column is an output activation "image"
    corresponding to the data input).

    Once you have implemented the autoencoder_cost_and_grad() function,
    simply copy the part of the code that computes the feedforward activations
    up to the output visible layer activations and return that activation.
    You do not need to include any of the computation of cost or gradient.
    It is likely that your implementation of feedforward in your
    autoencoder_cost_and_grad() is set up to handle multiple data inputs,
    in which case your only task is to ensure the output_activations matrix
    is in the same corresponding format as the input data matrix, where
    each output column is the activation corresponding to the input column
    of the same column index.

    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param data: input data matrix, where each column is an image patch,
                  with one or more columns
    :return: output_activations: an matrix output, where each column is the
                  vector of activations corresponding to the input data columns
    """
    m   = data.shape[1]
    len = visible_size * hidden_size
    
    """
    # DEBUG NUMPY RESHAPE ISSUES.
    print("m:",m)
    print("len:",len)
    print("theta:", theta)
    print("w2 reshape:", theta[len : 2*len].reshape((visible_size, hidden_size)) )
    """
                                                                         # Dimensions
    w1 = theta[0 : len].reshape((hidden_size, visible_size))             # (h,v)
    w2 = theta[len : 2*len].reshape((visible_size, hidden_size))         # (v,h)
    b1 = theta[2*len : 2*len + hidden_size].flatten()                    # (h)
    b2 = theta[2*len + hidden_size: ].flatten()                          # (v)

    
    # FORWARD PROPAGATION (Lecture 24, Slides 11-13)
    #
    # z^{l+1} = W^(l) * a^(l) + b^(l)  &&  a^{l+1} = h[z^(l+1)]
    #
    #   where
    #       z_i^(l) = Sum{j=1...n} W){ij}^{l-1} x_j + b_i^{l-1}
    #       a_i^(l) = h[z_i^(l)] = Activation (Output) of Unit i in Layer l
    
    z2 =  w1.dot(data) + np.repeat(b1,m).reshape((b1.shape[0],m))       # (h,m)
    a2 =  sigmoid(z2)                                                   # (h,m)
    
    z3 =  w2.dot(a2) + np.repeat(b2,m).reshape((b2.shape[0],m))         # (v,m)
    h  =  sigmoid(z3)                                                   # (v,m)


    return np.concatenate((z2, a2, z3, h))


# -------------------------------------------------------------------------

def save_model(theta, visible_size, hidden_size, filepath, **params):
    """
    Save the model to file.  Used by plot_and_save_results.
    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param filepath: path with filename
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """
    np.savetxt(filepath + '_theta.csv', theta, delimiter=',')
    with open(filepath + '_params.txt', 'a') as fout:
        params['visible_size'] = visible_size
        params['hidden_size'] = hidden_size
        fout.write('{0}'.format(params))


# -------------------------------------------------------------------------

def plot_and_save_results(theta, visible_size, hidden_size, root_filepath=None,
                          train_patches=None, test_patches=None, show_p=False,
                          **params):
    """
    This is a helper function to streamline saving the results of an autoencoder.
    The visible_size and hidden_size provide the information needed to retrieve
    the autoencoder parameters (w1, w2, b1, b2) from theta.

    This function does the following:
    (1) Saves the parameters theta, visible_size and hidden_size as a text file
        called '<root_filepath>_model.txt'
    (2) Extracts the layer 1 (input-to-hidden) weights and plots them as an image
        and saves the image to file '<root_filepath>_weights.png'
    (3) [optional] train_patches are intended to be a set of patches that were
        used during training of the autoencoder.  Typically these will be the first
        100 patches of the MNIST data set.
        If provided, the patches will be given as input to the autoencoder in
        order to generate output 'decoded' activations that are then plotted as
        patches in an image.  The image is saved to '<root_filepath>_train_decode.png'
    (4) [optional] test_patches are intended to be a different set of patches
        that were *not* used during training.  This permits inspecting how the
        autoencoder does decoding images it was not trained on.  The output activation
        image is generated the same way as in step (3).  The image is saved to
        '<root_filepath>_test_decode.png'

    The root_filepath is used as the base filepath name for all files generated
    by this function.  For example, if you wish to save all of the results
    using the root_filepath='results1', and you have specified the train_patches and
    test_patches, then the following files will be generated:
        results1_model.txt
        results1_weights.png
        results1_train_decode.png
        results1_test_decode.png
    If no root_filepath is provided, then the output will default to:
        model.txt
        weights.png
        train_decode.png
        test_decode.png
    Note that if those files already existed, they will be overwritten.

    :param theta: model parameters
    :param visible_size: number of nodes in autoencoder visible layer
    :param hidden_size: number of nodes in autoencoder hidden layer
    :param root_filepath: base filepath name for files generated by this function
    :param train_patches: matrix of patches (typically the first 100 patches of MNIST)
    :param test_patches: matrix of patches (intended to be patches NOT used in training)
    :param show_p: flag specifying whether to show the plots before exiting
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """

    filepath = 'model'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    save_model(theta, visible_size, hidden_size, filepath, **params)

    # extract the input to hidden layer weights and visualize all the weights
    # corresponding to each hidden node
    w1 = theta[0:hidden_size * visible_size].reshape((hidden_size, visible_size)).T
    filepath = 'weights.png'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    visualize.plot_images(w1, show_p=False, filepath=filepath)

    if train_patches is not None:
        # Given: train_patches and autoencoder parameters,
        # compute the output activations for each input, and plot the resulting decoded
        # output patches in a grid.
        # You can then manually compare them (visually) to the original input train_patches
        filepath = 'train_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        train_decode = autoencoder_feedforward(theta, visible_size, hidden_size, train_patches)
        visualize.plot_images(train_decode, show_p=False, filepath=filepath)

    if test_patches is not None:
        # Same as for train_patches, but assuming test_patches are patches that were not
        # used for training the autoencoder.
        # Again, you can then manually compare the decoded patches to the test_patches
        # given as input.
        test_decode = autoencoder_feedforward(theta, visible_size, hidden_size, test_patches)
        filepath = 'test_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        visualize.plot_images(test_decode, show_p=False, filepath=filepath)

    if show_p:
        plt.show()


# -------------------------------------------------------------------------

def get_pretty_time_string(t, delta=False):
    """
    Really cheesy kludge for producing semi-human-readable string representation of time
    y = Year, m = Month, d = Day, h = Hour, m (2nd) = minute, s = second, mu = microsecond
    :param t: datetime object
    :param delta: flag indicating whether t is a timedelta object
    :return:
    """
    if delta:
        days = t.days
        hours = t.seconds // 3600
        minutes = (t.seconds // 60) % 60
        seconds = t.seconds - (minutes * 60)
        return 'days={days:02d}, h={hour:02d}, m={minute:02d}, s={second:02d}' \
                .format(days=days, hour=hours, minute=minutes, second=seconds)
    else:
        return 'y={year:04d},m={month:02d},d={day:02d},h={hour:02d},m={minute:02d},s={second:02d},mu={micro:06d}' \
                .format(year=t.year, month=t.month, day=t.day,
                        hour=t.hour, minute=t.minute, second=t.second,
                        micro=t.microsecond)


# -------------------------------------------------------------------------
