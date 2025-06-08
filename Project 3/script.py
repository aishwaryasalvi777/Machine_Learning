import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.hstack((np.ones((n_data, 1)), train_data))
    w = initialWeights.reshape(-1, 1)
    z = np.dot(X, w)
    theta = sigmoid(z)

    error = (-1/n_data) * np.sum(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))
    error_grad = (1/n_data) * np.dot(X.T, (theta - labeli))
    error_grad = error_grad.flatten()
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.hstack((np.ones((data.shape[0], 1)), data))
    probabilities = sigmoid(np.dot(X, W))
    label = np.argmax(probabilities, axis=1).reshape(-1, 1)
    return label


def mlrObjFunction(params, *args):
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    X = np.hstack((np.ones((n_data, 1)), train_data))
    W = params.reshape((n_feature + 1, n_class))
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    error = (-1/n_data) * np.sum(labeli * np.log(probs + 1e-15))
    error_grad = (1/n_data) * np.dot(X.T, (probs - labeli))
    error_grad = error_grad.flatten()  
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.hstack((np.ones((data.shape[0], 1)), data))


    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    label = np.argmax(probs, axis=1).reshape(-1, 1)
    return label



"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


"""
Script for Support Vector Machine
"""
print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #


print("Using Linear Kernel (other parameters default)")
linear_svm = SVC(kernel='linear')
linear_svm.fit(train_data, train_label.ravel())


linear_train = accuracy_score(train_label, linear_svm.predict(train_data))
linear_val = accuracy_score(validation_label, linear_svm.predict(validation_data))
linear_test = accuracy_score(test_label, linear_svm.predict(test_data))

print(f"Training set Accuracy: {linear_train:.2%}")
print(f"Validation set Accuracy: {linear_val:.2%}")
print(f"Testing set Accuracy: {linear_test:.2%}")

print("\n---------------------------------\n") ## separator for understanding


print("RBF kernel with gamma value = 1")
rbf_gamma1 = SVC(kernel='rbf', gamma=1)
rbf_gamma1.fit(train_data, train_label.ravel())

rbf1_train = accuracy_score(train_label, rbf_gamma1.predict(train_data))
rbf1_val = accuracy_score(validation_label, rbf_gamma1.predict(validation_data))
rbf1_test = accuracy_score(test_label, rbf_gamma1.predict(test_data))

print(f"Training set Accuracy: {rbf1_train:.2%}")
print(f"Validation set Accuracy: {rbf1_val:.2%}")
print(f"Testing set Accuracy: {rbf1_test:.2%}")

print("\n---------------------------------\n")


print(" RBF kernel with  gamma value = default")
rbf_default = SVC(kernel='rbf')
rbf_default.fit(train_data, train_label.ravel())

rbfd_train = accuracy_score(train_label, rbf_default.predict(train_data))
rbfd_val = accuracy_score(validation_label, rbf_default.predict(validation_data))
rbfd_test = accuracy_score(test_label, rbf_default.predict(test_data))

print(f"Training set Accuracy: {rbfd_train:.2%}")
print(f"Validation set Accuracy: {rbfd_val:.2%}")
print(f"Testing set Accuracy: {rbfd_test:.2%}")

print("\n---------------------------------\n")


print("RBF  default gamma and varying C values")
C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_acc = []
val_acc = []
test_acc = []

for C in C_values:
    model = SVC(kernel='rbf', C=C)
    model.fit(train_data, train_label.ravel())
    
    train_acc.append(accuracy_score(train_label, model.predict(train_data)))
    val_acc.append(accuracy_score(validation_label, model.predict(validation_data)))
    test_acc.append(accuracy_score(test_label, model.predict(test_data)))
    
    print(f"C={C}: Train={train_acc[-1]:.2%}, Val={val_acc[-1]:.2%}, Test={test_acc[-1]:.2%}")


plt.figure(figsize=(10, 6))
plt.plot(C_values, train_acc, label='Training Accuracy', marker='o')
plt.plot(C_values, val_acc, label='Validation Accuracy', marker='s')
plt.plot(C_values, test_acc, label='Testing Accuracy', marker='^')
plt.xlabel('C Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C Value (RBF Kernel)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy.png')
plt.close()
##################

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1) * n_class)
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
