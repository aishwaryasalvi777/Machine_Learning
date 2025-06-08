import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
from datetime import timedelta
import time
import itertools
import matplotlib.pyplot as plt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return   1/(1+np.exp(-z))
# 



def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    # Load the MNIST dataset
    mat = loadmat('mnist_all.mat') 

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    
    # Initialize preprocess predefined NumPy arrays 
    train_preprocess = np.zeros(shape=(50000, 784))       # Training data: 50,000 samples.
    validation_preprocess = np.zeros(shape=(10000, 784))  # Validation data: 10,000 samples
    test_preprocess = np.zeros(shape=(10000, 784))        # Test data: 10,000 samples 
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))

    # flag variables to track current positions in arrays
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0

    # Iteration to split the data set
    for key in mat:
        if "train" in key:
            label = key[-1]  
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  
            tag_len = tup_len - 1000  
            # Approximately 80%  of examples added into the training set

            # Data added to training set
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # Data added to validation set
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000 

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
    
    # Shuffle
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    # Convert data to double type for numerical stability
    train_data = train_data / 255.0
    # Normalize pixel values to the range [0, 1]
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
  
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]
    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection                     # removal of features(pixels)
                                            # with low var
    var_feature = np.var(train_data, axis=0)  # variance calculation
    selected = np.where(var_feature > 0.1)[0] 
    train_data = train_data[:, selected]
    test_data = test_data[:, selected]
    validation_data = validation_data[:, selected]

    print("PreProcessing Done!!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Feedforward propagation
    n_samples = training_data.shape[0]
    bias_input = np.ones((n_samples, 1))
    input_data = np.concatenate((training_data, bias_input), axis=1)

    hidden_layer_input = np.dot(input_data, w1.T)
    hidden_layer_output = sigmoid(hidden_layer_input)

    bias_hidden = np.ones((n_samples, 1))
    hidden_layer_output = np.concatenate((hidden_layer_output, bias_hidden), axis=1)

    output_layer_input = np.dot(hidden_layer_output, w2.T)
    output_layer_output = sigmoid(output_layer_input)
    # applied sigmoid activation function to get output 

    # Error function implemtation
    y = np.zeros((n_samples, n_class))
    np.add.at(y, (np.arange(n_samples), training_label.astype(int)), 1)

    error_function = -(1 / n_samples) * np.sum(y * np.log(output_layer_output) + (1 - y) * np.log(1 - output_layer_output))
    
    # regularization term implemetation for overfitting prevention
    regularization = (lambdaval / (2 * n_samples)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error_function + regularization

    # Backpropagation for gradient computation 
    delta_output = output_layer_output - y

    grad_w2 = (1 / n_samples) * np.dot(delta_output.T, hidden_layer_output)
    # Addng regularization term
    grad_w2[:, :-1] += (lambdaval / n_samples) * w2[:, :-1]

    delta_hidden = np.dot(delta_output, w2[:, :-1]) * hidden_layer_output[:, :-1] * (1 - hidden_layer_output[:, :-1])
    grad_w1 = (1 / n_samples) * np.dot(delta_hidden.T, input_data)
    grad_w1[:, :-1] += (lambdaval / n_samples) * w1[:, :-1]

    # Flatten gradients to convert into single 1D array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)



def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    # Initialize labels as an empty array
    labels = np.array([])

    n_samples = data.shape[0]

    # Added bias to the input terms 
    bias_input = np.ones((n_samples, 1))
    input_with_bias = np.concatenate((data, bias_input), axis=1)

    # Calculate hidden layer output
    hidden_layer_input = np.dot(input_with_bias, w1.T)
    hidden_layer_output = sigmoid(hidden_layer_input)
    # sigmoid actiavtion finction activation .

    # Add bias to the hidden layer output terms 
    bias_hidden = np.ones((n_samples, 1))
    hidden_with_bias = np.concatenate((hidden_layer_output, bias_hidden), axis=1)

    # Calculate output layer output
    output_layer_input = np.dot(hidden_with_bias, w2.T)
    output_layer_output = sigmoid(output_layer_input)


    # Predict labels by selecting highest probability 
    labels = np.argmax(output_layer_output, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

if __name__ == "__main__":
    # Data Preprocessing  functio call :preprocess
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # Number of nodes initialisation 
    n_input = train_data.shape[1]

    # Setting nodes in hidden unit 
    n_hidden = 50

    # Setting nodes in output unit
    n_class = 10

    # Initialization the weights 
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # Setting the regularization parameter(lambda value)
    lambdaval = 0
    #lambdaval = 0.01
    #lambdaval = 0.05

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

 
    opts = {'maxiter': 50} 
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)


    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    predicted_label = nnPredict(w1, w2, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    # Hyperparameter tuning 
    lambda_values = list(range(1, 62, 10))
    hidden_units_list = [10, 20, 30, 40, 50]

    results = {'lambda': [], 'hidden_units': [], 'train_accuracy': [], 'validation_accuracy': [], 'test_accuracy': [], 'training_time': []}

    best_validation_accuracy = 0
    best_params = None
    selected_params = None

    for lambdaval, n_hidden in itertools.product(lambda_values, hidden_units_list):
        start_time = time.time()  

        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        opts = {'maxiter': 50}

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        training_time = time.time() - start_time

        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        train_accuracy = 100 * np.mean(nnPredict(w1, w2, train_data) == train_label)
        validation_accuracy = 100 * np.mean(nnPredict(w1, w2, validation_data) == validation_label)
        test_accuracy = 100 * np.mean(nnPredict(w1, w2, test_data) == test_label)

        results['lambda'].append(lambdaval)
        results['hidden_units'].append(n_hidden)
        results['train_accuracy'].append(train_accuracy)
        results['validation_accuracy'].append(validation_accuracy)
        results['test_accuracy'].append(test_accuracy)
        results['training_time'].append(training_time)

        # Best model based on validation accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_params = {'lambda': lambdaval, 'hidden_units': n_hidden, 'train_accuracy': train_accuracy,
                           'validation_accuracy': validation_accuracy, 'test_accuracy': test_accuracy,
                           'training_time': training_time, 'w1': w1, 'w2': w2}
            selected_params = {'hidden_units': n_hidden, 'w1': w1, 'w2': w2, 'lambda': lambdaval  }

    # Best model 
    with open('params_nnScript.pickle', 'wb') as f:
        pickle.dump(selected_params, f)


    # Results Plotting
    plt.figure(figsize=(12, 8))
    for hidden_units in hidden_units_list:
        mask = np.array(results['hidden_units']) == hidden_units
        plt.plot(np.array(results['lambda'])[mask], np.array(results['validation_accuracy'])[mask], label=f'Hidden Units={hidden_units}')
    plt.xlabel('Lambda Value')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy vs Lambda Value for different Hidden Units')
    plt.legend()
    plt.grid(True)
    plt.savefig('lambda_vs_Validation_accuracy_hidden_units.png')
    plt.close()

    # Plotting the results for test accuracy
    plt.figure(figsize=(12, 8))
    for hidden_units in hidden_units_list:
        mask = np.array(results['hidden_units']) == hidden_units
        plt.plot(np.array(results['lambda'])[mask], np.array(results['test_accuracy'])[mask], label=f'Hidden Units={hidden_units}')
    plt.xlabel('Lambda Value')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Lambda Value for different Hidden Units')
    plt.legend()
    plt.grid(True)
    plt.savefig('lambda_vs_Test_accuracy_hidden_units.png')
    plt.close()

    # Plotting the average training time vs number of hidden units
    unique_hidden_units = np.unique(results['hidden_units'])
    mean_training_times = []

    for hidden_units in unique_hidden_units:
        mask = np.array(results['hidden_units']) == hidden_units
        mean_training_time = np.mean(np.array(results['training_time'])[mask])
        mean_training_times.append(mean_training_time)

    plt.figure(figsize=(10, 6))
    plt.plot(unique_hidden_units, mean_training_times, marker='o')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Average Training Time')
    plt.title('Average Training Time vs Number of Hidden Units')
    plt.grid(True)
    plt.savefig('average_training_time_vs_hidden_units.png')
    plt.close()