'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt
import pickle
from datetime import timedelta
import time
import itertools
import matplotlib.pyplot as plt


# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
        return   1/(1+np.exp(-z))

    
    
# Replace this with your nnObjFunction implementation

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

    # Compute error function
    y = np.zeros((n_samples, n_class))
    np.add.at(y, (np.arange(n_samples), training_label.astype(int)), 1)

    error_function = -(1 / n_samples) * np.sum(y * np.log(output_layer_output) + (1 - y) * np.log(1 - output_layer_output))
     #---

    # Add regularization term
    regularization = (lambdaval / (2 * n_samples)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error_function + regularization

    # Backpropagation
    delta_output = output_layer_output - y
    grad_w2 = (1 / n_samples) * np.dot(delta_output.T, hidden_layer_output)
    grad_w2[:, :-1] += (lambdaval / n_samples) * w2[:, :-1]

    delta_hidden = np.dot(delta_output, w2[:, :-1]) * hidden_layer_output[:, :-1] * (1 - hidden_layer_output[:, :-1])
    grad_w1 = (1 / n_samples) * np.dot(delta_hidden.T, input_data)
    grad_w1[:, :-1] += (lambdaval / n_samples) * w1[:, :-1]

    # Flatten gradients
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)

     
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    # Initialize labels as an empty array
    labels = np.array([])

    # Your code here
    n_samples = data.shape[0]

    # Add bias term to the input data
    bias_input = np.ones((n_samples, 1))
    input_with_bias = np.concatenate((data, bias_input), axis=1)

    # Compute hidden layer output
    hidden_layer_input = np.dot(input_with_bias, w1.T)
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Add bias term to the hidden layer output
    bias_hidden = np.ones((n_samples, 1))
    hidden_with_bias = np.concatenate((hidden_layer_output, bias_hidden), axis=1)

    # Compute output layer output
    output_layer_input = np.dot(hidden_with_bias, w2.T)
    output_layer_output = sigmoid(output_layer_input)

    # Predict labels by taking the argmax of the output layer
    labels = np.argmax(output_layer_output, axis=1)

    return labels



# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit 
n_input = train_data.shape[1]
# set the number of nodes in hidden unit 
n_hidden = 256
# set the number of nodes in output unit
n_class = 2


initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

lambdaval = 10
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')


#hidden_units_list = [4, 8, 12, 16, 20]
#same landa =10


# Hyperparameter tuning with constant lambda and varying hidden units
lambda_val = 10  # Constant lambda value
hidden_units_list = [4, 8, 12, 16, 20]  # List of hidden units to test

# Dictionary to store results
results = {'hidden_units': [], 'train_accuracy': [], 'validation_accuracy': [], 'test_accuracy': [], 'training_time': []}

# Best model tracking
best_validation_accuracy = 0
best_params = None

# Iterate over hidden units
for n_hidden in hidden_units_list:
    start_time = time.time()  # Start timer

    # Initialize weights
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # Prepare arguments for training
    args = (n_input, n_hidden, n_class, train_data, train_label, lambda_val)
    opts = {'maxiter': 50}  # Maximum number of iterations

    # Train Neural Network
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    training_time = time.time() - start_time  # Calculate training time

    # Reshape trained weights
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Calculate accuracies
    train_accuracy = 100 * np.mean(nnPredict(w1, w2, train_data) == train_label)
    validation_accuracy = 100 * np.mean(nnPredict(w1, w2, validation_data) == validation_label)
    test_accuracy = 100 * np.mean(nnPredict(w1, w2, test_data) == test_label)

    # Store results
    results['hidden_units'].append(n_hidden)
    results['train_accuracy'].append(train_accuracy)
    results['validation_accuracy'].append(validation_accuracy)
    results['test_accuracy'].append(test_accuracy)
    results['training_time'].append(training_time)

    # Track the best model based on validation accuracy
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_params = {
            'hidden_units': n_hidden,
            'w1': w1,
            'w2': w2,
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy,
            'test_accuracy': test_accuracy,
            'training_time': training_time
        }

    # Print results for current configuration
    print(f'\nHidden Units: {n_hidden}')
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Accuracy: {validation_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Training Time: {training_time:.2f} seconds')

# plot ==
#
#
# Plotting the average training time vs number of hidden units
plt.figure(figsize=(10, 6))
plt.plot(results['hidden_units'], results['training_time'], marker='o')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Average Training Time (seconds)')
plt.title('Average Training Time vs Number of Hidden Units')
plt.grid(True)
plt.savefig('average_training_time_vs_hidden_units.png')
plt.close()