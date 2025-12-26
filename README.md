# Machine Learning Projects Collection

A comprehensive collection of three machine learning projects implementing various neural network architectures and classification algorithms. These projects cover fundamental to advanced deep learning techniques using PyTorch, TensorFlow, and scikit-learn.

---

## 📁 Project Structure

```
Machine_Learning/
├── Project 1/           # Deep Learning with PyTorch & Custom Neural Networks
│   ├── Code/
│   │   ├── cnnScript.py           # Convolutional Neural Network (CNN)
│   │   ├── deepnnScript.py        # Deep Multi-layer Perceptron (PyTorch)
│   │   ├── facennScript.py        # Face Recognition Neural Network (TensorFlow)
│   │   ├── nnScript.py            # Multi-layer Perceptron (NumPy/SciPy)
│   │   ├── face_all.pickle        # Face dataset
│   │   └── params_nnScript.pickle # Trained model parameters
│   └── Video/
│       └── ML project1.mp4        # Project documentation video
├── Project 2/           # Linear & Quadratic Discriminant Analysis
│   └── script.py                  # LDA/QDA implementation
└── Project 3/           # Support Vector Machines & Logistic Regression
    └── script.py                  # SVM/Logistic Regression implementation
```

---

## 🚀 Project Descriptions

### **Project 1: Deep Learning with Neural Networks**

A comprehensive exploration of neural network architectures for image classification and face recognition tasks.

#### Components:

**1. `nnScript.py` - Multi-layer Perceptron (MLP)**
- **Dataset**: MNIST (handwritten digit recognition)
- **Architecture**: Custom implementation using NumPy and SciPy
- **Features**:
  - Weight initialization with Xavier initialization
  - Sigmoid activation function
  - Backpropagation algorithm
  - Optimization using scipy.optimize.minimize
- **Functionality**: Trains a neural network from scratch to classify MNIST digits (0-9)

**2. `deepnnScript.py` - Deep Multi-layer Perceptron (PyTorch)**
- **Dataset**: Face recognition dataset (`face_all.pickle`)
- **Architecture**: 
  - Input layer: 2,376 features
  - 3 Hidden layers: 256 neurons each with ReLU activation
  - Output layer: 2 classes (binary classification)
- **Framework**: PyTorch with GPU support
- **Features**:
  - Modern deep learning framework implementation
  - Batch processing with DataLoader
  - Train/validation/test split (21,100/2,665/2,670 samples)
  - Cross-entropy loss function

**3. `cnnScript.py` - Convolutional Neural Network (CNN)**
- **Dataset**: MNIST
- **Architecture**:
  - Conv Layer 1: 16 filters of 5×5 pixels
  - Conv Layer 2: 36 filters of 5×5 pixels
  - Max Pooling layers (2×2)
  - Fully Connected Layer 1: 128 neurons
  - Output Layer: 10 classes
- **Framework**: PyTorch
- **Features**:
  - Convolutional feature extraction
  - Max pooling for dimensionality reduction
  - Visualization of predictions on test images
  - Confusion matrix analysis

**4. `facennScript.py` - Face Recognition Neural Network**
- **Dataset**: Face recognition dataset (`face_all.pickle`)
- **Architecture**: 2-layer neural network
  - Input features: 2,376 (facial features)
  - Hidden layer: Configurable (typically 256)
  - Output: Binary classification
- **Framework**: TensorFlow/NumPy implementation
- **Features**:
  - Sigmoid activation functions
  - Cross-entropy loss with regularization (L2)
  - Gradient descent optimization
  - Weight initialization with epsilon bounds

---

### **Project 2: Statistical Classification Methods**

Implementation of Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) for multiclass classification.

#### Key Functions:

**`ldaLearn(X, y)`**
- Computes class means for each class
- Calculates pooled covariance matrix across all classes
- Assumes equal covariance matrices for all classes

**`qdaLearn(X, y)`**
- Computes class-specific means
- Learns separate covariance matrix for each class
- More flexible than LDA for data with class-dependent variance

**`ldaTest(means, covmat, Xtest, ytest)`**
- Applies LDA discriminant function for classification
- Computes accuracy on test set
- Returns predicted labels and accuracy

**`qdaTest(means, covmats, Xtest, ytest)`**
- Applies QDA discriminant function for classification
- Uses class-specific covariance matrices
- Similar output to ldaTest

#### Dataset:
- **MNIST** digits (10 classes)
- Multi-class classification task
- Training/validation/test split included

---

### **Project 3: Modern Classification Algorithms**

Comprehensive implementation of Support Vector Machines (SVM) and Logistic Regression for MNIST digit classification.

#### Key Components:

**Preprocessing (`preprocess()`)**
- Loads MNIST dataset from `mnist_all.mat`
- Splits into training (9,000), validation (10,000), and test (10,000) sets
- Feature selection based on variance (removes low-variance features)
- Normalizes data to [0, 1] range

**Logistic Regression**
- Two-class logistic regression (`blrObjFunction`)
- Computes sigmoid function, error, and gradient
- Uses scipy.optimize.minimize for optimization
- Supports regularization (L2)

**Support Vector Machine (SVM)**
- Multi-class SVM using scikit-learn
- Configurable kernel (linear, RBF, polynomial)
- Hyperparameter tuning
- Uses sklearn.svm.SVC for implementation

#### Dataset:
- **MNIST** handwritten digits
- 10 classes (digits 0-9)
- Preprocessed and feature-selected

---

## 📋 Dependencies

### Core Libraries:
- **NumPy**: Numerical computations and linear algebra
- **SciPy**: Optimization and scientific functions
- **PyTorch**: Deep learning framework (Project 1)
- **TensorFlow/Keras**: Deep learning (Project 1)
- **scikit-learn**: Machine learning algorithms (Project 2 & 3)
- **Matplotlib**: Data visualization
- **pickle**: Data serialization

### Installation:
```bash
pip install numpy scipy torch torchvision tensorflow scikit-learn matplotlib
```

---

## 🎯 Use Cases

| Project | Algorithm | Use Case | Dataset |
|---------|-----------|----------|---------|
| **Project 1** | MLP, CNN, Deep NN | Image classification, Face recognition | MNIST, Face recognition |
| **Project 2** | LDA, QDA | Statistical classification, Dimensionality reduction | MNIST |
| **Project 3** | SVM, Logistic Regression | Binary/Multi-class classification | MNIST |

---

## 🔧 Running the Projects

### Project 1: Neural Networks

**MNIST Classification with MLP:**
```bash
python Project\ 1/Code/nnScript.py
```

**MNIST Classification with CNN:**
```bash
python Project\ 1/Code/cnnScript.py
```

**Face Recognition with Deep NN:**
```bash
python Project\ 1/Code/deepnnScript.py
# Requires face_all.pickle in the same directory
```

**Face Recognition with TensorFlow:**
```bash
python Project\ 1/Code/facennScript.py
# Requires face_all.pickle in the same directory
```

### Project 2: LDA/QDA
```bash
python Project\ 2/script.py
# Requires mnist_all.mat
```

### Project 3: SVM/Logistic Regression
```bash
python Project\ 3/script.py
# Requires mnist_all.mat
```

---

## 📊 Key Algorithms Explained

### Neural Networks (Project 1)
- **Backpropagation**: Gradient-based optimization for updating weights
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)
- **Convolutional Layers**: Feature extraction for spatial data
- **Max Pooling**: Dimensionality reduction and translation invariance

### LDA & QDA (Project 2)
- **LDA**: Assumes Gaussian distribution with shared covariance
- **QDA**: Assumes Gaussian distribution with class-specific covariance
- **Discriminant Function**: Linear/Quadratic boundary for class separation

### SVM & Logistic Regression (Project 3)
- **SVM**: Finds maximum-margin hyperplane for classification
- **Logistic Regression**: Probability-based linear classification
- **Feature Selection**: Removes low-variance features to improve performance

---

## 📈 Performance Metrics

All projects include evaluation metrics:
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Class-wise prediction breakdown
- **Loss Functions**: Cross-entropy, MSE, Hinge loss (depending on algorithm)
- **Training/Validation/Test Split**: Separate evaluation phases

---

## 📹 Documentation

Project 1 includes a video documentation file (`ML project1.mp4`) explaining the implementation and results.

---

## 📝 Notes

- **Data Files**: Some scripts require external data files (mnist_all.mat, face_all.pickle)
- **GPU Support**: Project 1 PyTorch scripts can utilize GPU for faster training
- **Hyperparameters**: Tunable parameters available in each script for experimentation
- **Regularization**: L2 regularization available to prevent overfitting

---

## 🎓 Learning Outcomes

These projects demonstrate proficiency in:
✅ Neural network architecture design and implementation  
✅ Convolutional neural networks for image processing  
✅ Deep learning frameworks (PyTorch, TensorFlow)  
✅ Classical statistical ML methods (LDA, QDA)  
✅ Modern ML algorithms (SVM, Logistic Regression)  
✅ Data preprocessing and feature engineering  
✅ Model evaluation and performance analysis  
✅ Optimization algorithms (Gradient descent, scipy minimize)  

---

## 📄 License

University coursework project from University at Buffalo

---

**Last Updated**: December 2025
