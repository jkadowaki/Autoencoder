from __future__ import print_function
import load_MNIST
import visualize
import utils
import gradient
import numpy
import scipy.optimize
import datetime


# ======================================================================
# Top-level flags to control execution of portions of the script
# Once steps 1-4 are completed, they can be set to False while
# RUN_STEP_5_TRAIN_AUTOENCODER is True.

RUN_STEP_1 = False  # visualize patches, exercise 1
RUN_STEP_2 = False  # implement utils.initialize, exercise 2
RUN_STEP_3 = False  # implement utils.autoencoder_cost_and_grad{_sparse}, exercise 3 (first part), and revisit with exercise 6
RUN_STEP_4_DEBUG_GRADIENT = False  # implement gradient.compute_gradient_numerical_estimate, exercise 3 (second), and revisit with exercise 6
RUN_STEP_5_TRAIN_AUTOENCODER = True  # exercise 5, exercise 6

# ======================================================================
# Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to achieve good filters;
#  you do not need to change the parameters below.

# number of input units
visible_size = 28 * 28
# number of hidden units
hidden_size = 100

beta_ = 0.1
rho_ = 0.05
#rho_ = [0.05, 0.01, 0.005]

# desired average activation of the hidden units.
# (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
#  in the lecture notes).

# weight decay (weight regularization) parameter
lambda_ = 0.0001


# ======================================================================
# Step 0: Load MNIST and visualize
# In this exercise, you will load the mnist dataset
# First download the dataset from the following website:
# Training Images: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# Training Labels: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# Loading Sample Images
# Loading 60K images from MNIST database

# NOTE: YOU LIKELY NEED TO CHANGE THE FOLLOWING PATH TO WHERE YOU STORED THE MNIST DATA
images = load_MNIST.load_MNIST_images('../data/train-images-idx3-ubyte')
# Each column represents one 28x28 pixel image (784 total pixels) that has
# been "unrolled" into a 784-element column vector.
patches_train = images[:, 0:100]  # grabs the first 100 images (i.e., the first 100 columns)
patches_test = images[:, 1200:1300]  # grabs 100 image patches that will be used for 'testing'



# ======================================================================
# STEP 1: Visualize patches_train
# You will now use the visualize.py function plot_images() to display different
# sets if the MNIST dataset.  You can provide a filename argument to plot_images
# which will specify the name of the file the image is saved as;
# default is to save the file in the directory under the name 'weights.png'.
# Your task:
#     Plot the first 10, 50 and 100 images of patches_train
#     Also plot the first 100 image patches that will be used for 'testing' (patches 1200 to 1300)

if RUN_STEP_1:
    
    for num in [10,50,100]:
        visualize.plot_images(patches_train[:, 0:num],
                              filepath='patches_train{:03d}.png'.format(num))
    
    visualize.plot_images(patches_test[:, 0:100], filepath='patches_test100.png')


# ======================================================================
# STEP 2: Implement utils.initialize
# Obtain random parameters theta ; see Exercise 2
# You need to implement initialize in utils.py
if RUN_STEP_2:
                          
    theta = utils.initialize(hidden_size, visible_size)


# ======================================================================
# STEP 3: Implement mlp_cost_and_grad
#
#  In this step you will implement the calculation of the loss (cost)
#  and theta gradient (grad) in utils.autoencoder_cost_and_grad.
#
#  You can implement all of the components in the cost function at once,
#  but it may be easier to do it step-by-step and run gradient checking
#  (see STEP 4) after each step.  We suggest implementing the
#  utils.autoencoder_cost_and_grad function using the following steps:
#
#  (a) Implement forward propagation in your neural network, and implement
#      the regularized loss function described in the tutorial and
#      class lectures.  Implement backpropagation to compute the derivatives.
#      Then (using lambda=beta=0), run Gradient Checking (i.e.,
#      gradient.compute_gradient_numerical_estimate) to verify that the
#      calculations of the gradient in utils.autoencoder_cost_and_grad
#      are sufficiently close to the numerical gradient estimate.
#
#  (b) Add in the weight decay term (in both the cost function and the
#      derivative calculations), then re-run Gradient Checking to verify
#      correctness.
#

#  Feel free to change the training settings when debugging your
#  code_solution.  For example, reducing the training set size and/or
#  the number of hidden units may make your code run faster; and
#  setting lambda_ to zero may be helpful for debugging.

#  Finally, when you get to step TODO and implement
#  utils.autoencoder_cost_and_grad_sparse, you can test it here and
#  using the Step 4 debug as well.

if RUN_STEP_3:
    (cost, grad) = utils.autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, patches_train)
    #(cost, grad) = utils.autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, patches_train)

# ======================================================================
# STEP 4: Gradient Checking
#
# Hint: If you are debugging your code_solution, performing gradient
# checking on smaller models and smaller training sets (e.g., using only
# 10 training examples and 1-2 hidden units) will speed things up.

# First, make sure your numerical gradient computation is correct
# for a simple function.

# After you have implemented gradient.compute_gradient_numerical_estimate,
# run this script while setting DEBUG = True on line 30 above.
# Once you are convinced your implementation of the cost and gradient
# calculation are correct, you can set DEBUG back to False (as when you
# return to realistic hidden unit and training data sizes, the gradient
# checking will get very slow!

if RUN_STEP_4_DEBUG_GRADIENT:
    print("========== DEBUG: checking gradient ==========")

    # the following test your implementation of compute_gradient_numerical_estimate
    gradient.test_compute_gradient_numerical_estimate()

    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
    # J is the cost function

    print('Now test autoencoder_cost_and_grad() gradient against numerical estimate:')
    print('    Total number of parameters, theta.shape= {0}'.format(theta.shape))

    # define the objective function that returns cost and grad, used by scipy.optimizze.minimize
    J = lambda x: utils.autoencoder_cost_and_grad_sparse(x, visible_size, hidden_size, lambda_, rho_, beta_, patches_train)
    #J = lambda x: utils.autoencoder_cost_and_grad(x, visible_size, hidden_size, lambda_, patches_train)
    num_grad = gradient.compute_gradient_numerical_estimate(J, theta)

    # The following is for optional additional debugging
    # Uncomment the following to look at the individual differences for each parameter.
    # Sometimes this can be informative.
    """
    for i in range(theta.shape[0]):
        if i == (visible_size * hidden_size) \
                or i == 2 * (visible_size * hidden_size)\
                or i == 2 * (visible_size * hidden_size) + hidden_size:
            print('=======================')
        #print('{0} : {1} {2} {3}'.format(i, num_grad[i] / 2 - grad[i], num_grad[i], grad[i]))
        if grad[i]-num_grad[i]>10**-9:
            print("{0}: {1} {2} {3}".format(i, num_grad[i], grad[i], grad[i]-num_grad[i]))
    """
    # Compare numerically computed gradients with the ones obtained from backpropagation
    diff = numpy.linalg.norm(num_grad - grad)
    print("Norm of the difference between numerical and autoencoder_cost_and_grad gradients:")
    print("    ", diff)
    print("(should be at least < 1.0e-07)")

    print("========== DEBUG: checking gradient DONE ==========")


# ======================================================================
# STEP 5: After verifying that your implementation of
#  utils.autoencoder_cost_and_grad is correct, You can start training your
#  autoencoder, using scipy.optimize.minimize L-BFGS-B.

if RUN_STEP_5_TRAIN_AUTOENCODER:
    #  Initialize the parameters
    theta = utils.initialize(hidden_size, visible_size)

    print("\nRunning scipy.optimize.minimize on {0} parameters, over {1} training patches_train"
          .format(theta.shape[0], patches_train.shape[1]))
    start_time = datetime.datetime.now()
    print("    START TIME {0}".format(utils.get_pretty_time_string(start_time)))
    # define the objective function that returns cost and grad, used by scipy.optimizze.minimize
    J = lambda x: utils.autoencoder_cost_and_grad_sparse(x, visible_size, hidden_size, lambda_, rho_, beta_, patches_train)
    #J = lambda x: utils.autoencoder_cost_and_grad(x, visible_size, hidden_size, lambda_, patches_train)
    options_ = {'maxiter': 4000, 'disp': False}
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    opt_theta = result.x  # theta found after optimization

    end_time = datetime.datetime.now()
    print("    END TIME {0}".format(utils.get_pretty_time_string(end_time)))
    total_time = end_time - start_time
    time_elapsed_string = utils.get_pretty_time_string(total_time, delta=True)
    print("    Total run time: {0}".format(time_elapsed_string))

    print("\nscipy.optimize.minimize() details:")
    print(result)

    print("\nNOTE: Don't worry if scipy.optimize.minimize() reports 'success: False'")
    print("      due to hitting the maximum iterations, or a message indicating an")
    print("      error condition.  (nit = 'number of iterations')")
    print("      I have set the max iterations to 4000, which is generally")
    print("      sufficient for our purposes here.")

    # Step 6: Visualize and save results

    results_filepath_root = 'autoencoder_k{0}_h{1}_l{2}'.format(patches_train.shape[1], hidden_size, lambda_)


    print("\n For Debugging",
          "\n\topt_theta:", opt_theta.shape,
          "\n\tvisible size:", visible_size,
          "\n\thidden size:", hidden_size,
          "\n\tfile:", results_filepath_root,
          "\n\tpatches_train shape:", patches_train[:, 0:100].shape,
          "\n\tpatches_test shape:", patches_test.shape,
          "\n\tlambda_:", lambda_,
          "\n\trho_:", rho_,
          "\n\tbeta_:", beta_,
          "\n\ttrain time:", time_elapsed_string,
          "\n\tnit:", result.nit)
    
    print(opt_theta)

    utils.plot_and_save_results\
        (opt_theta, visible_size, hidden_size,
         root_filepath=results_filepath_root,
         train_patches=patches_train[:, 0:100],  # only use the first 100 patches, or visualization may get messy
         test_patches=patches_test,
         show_p=True,
         # Everything after this point is stored as a dictionary in the 'params' argument:
         lambda_=lambda_,
         rho_=rho_,    # for when you implement utils.autoencoder_cost_and_grad_sparse
         beta_=beta_,  # for when you implement utils.autoencoder_cost_and_grad_sparse
         train_time=time_elapsed_string,
         nit=result.nit,
         success=result.success,
         message=result.message)

