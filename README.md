# MLScratchPad: Machine Learning from Scratch
MLScratchPad is a hands-on approach to building classic machine learning models from the ground up, focusing on algorithmic insights without the use of libraries like sklearn.

**Table of Contents**
1. [Introduction](#introduction)
2. [Models](#models)
3. [Features](#features)
4. [Prerequisites](#prerequisites)
5. [Usage](#usage)
   - [Getting Started](#getting-started)
   - [Using a Learner](#using-a-learner)
   - [Example Usage](#example-usage)
6. [License](#license)

## Introduction
Welcome to MLScratchPad, a hands-on repository where classic machine learning models are built from the ground up. This project is designed for those who want to understand the nitty-gritty of machine learning algorithms without relying on high-level libraries like scikit-learn. It's perfect for students, hobbyists, and professionals looking to deepen their understanding of the underlying mechanics of machine learning models.

## Models
- **Decision Tree Learner (`DTLearner.py`)**: Implements a basic decision tree learning algorithm.
- **Linear Regression Learner (`LinRegLearner.py`)**: A foundational model for linear regression analysis.
- **Random Tree Learner (`RTLearner.py`)**: Implements a random tree learning approach.
- **AdaBoost Learner (`AdaBoost.py`)**: *(Note: Currently under development)* An initial implementation of the AdaBoost algorithm for boosting weaker models. This module is not yet available as an object-oriented interface.
- **Bagging Learner (`BagLearner.py`)**: Introduces the concept of bagging in machine learning.
- **Gaussian Mixture Model Expectation-Maximization Learner (`GMMEMLearner.py`)**: Applies GMM-EM for clustering and density estimation.
- **Insane Learner (`InsaneLearner.py`)**: A complex learner that combines various algorithms for enhanced learning capabilities.
- **ISOMAP Learner (`ISOMAPLearner.py`)**: Focuses on dimensionality reduction using ISOMAP algorithm.
- **K-Means Learner (`KMeansLearner.py`)**: Implements the K-Means clustering algorithm.
- **PCA Learner (`PCALearner.py`)**: A learner for Principal Component Analysis (PCA) for dimensionality reduction.
- **Q-Learning Learner (`QLearner.py`)**: Demonstrates reinforcement learning using the Q-Learning algorithm.
- **Ridge Regression Learner (`RidgeRegLearner.py`)**: *(Note: Currently under development)* This module presents an initial implementation of Ridge Regression, which is yet to be converted into an object-oriented form.
- **Spectral Learner (`SpectralLearner.py`)**: Applies spectral clustering techniques for machine learning.

## Features
- **No High-level Libraries**: All algorithms are implemented without using libraries like sklearn, ensuring a deeper understanding of the algorithms.
- **Algorithmic Insights**: Emphasizes the algorithmic and mathematical principles underlying each model.
- **Code Documentation**: Each file is thoroughly documented, making it easy to follow and learn from the code.

## Prerequisites
- Basic knowledge of Python programming.
- Understanding of fundamental machine learning concepts.

## Usage
This section guides you through the process of using the machine learning models in MLScratchPad. 

### Getting Started
1. **Clone the Repository**: First, clone MLScratchPad to your local machine using Git:
   ```bash
   git clone https://github.com/yourusername/MLScratchPad.git
   ```
   Replace `yourusername` with your GitHub username or the URL of the repository.

2. **Navigate to the Repository**: Change your directory to the MLScratchPad folder:
   ```bash
   cd MLScratchPad
   ```

### Using a Learner
3. **Choose a Learner**: Decide which machine learning model you want to use. For example, if you want to use the Decision Tree Learner, you'll work with `DTLearner.py`.

4. **Prepare Your Python Environment**: Make sure you have Python installed. You can use virtual environments to manage your packages.

5. **Write Your Script**: In your Python environment, create a new Python script or open an interactive session. Import the learner class from the corresponding file. For example:
   ```python
   from DTLearner import DTLearner
   ```

6. **Initialize the Learner**: Create an instance of the learner. For instance, with `DTLearner`, you might do:
   ```python
   learner = DTLearner()
   ```

7. **Load Your Data**: Load the dataset you wish to train on. You can use any dataset in a compatible format (e.g., CSV, Excel, SQL database).

8. **Train the Learner**: Call the appropriate method to train your learner. For example:
   ```python
   learner.addEvidence(trainX, trainY)
   ```
   where `trainX` is your training input and `trainY` is the training output.

9. **Test the Learner**: Once trained, you can make predictions or evaluate the model using your test data.

### Example Usage
Here is a simple example of using a learner:
```python
# Example with Decision Tree Learner
from DTLearner import DTLearner
import numpy as np

# Create a learner instance
learner = DTLearner()

# Sample data
trainX = np.array([[1, 2], [3, 4]])
trainY = np.array([0.5, 1.5])

# Train the learner
learner.addEvidence(trainX, trainY)

# Make predictions
testX = np.array([[5, 6]])
predictions = learner.query(testX)
print(predictions)
```

Remember to consult the documentation in each learner file for specific instructions and parameters.
