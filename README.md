# MLScratchPad: Machine Learning from Scratch
MLScratchPad is a hands-on approach to building classic machine learning models from the ground up, focusing on algorithmic insights without the use of libraries like sklearn.

**Table of Contents**
1. [Introduction](#introduction)
2. [Models](#models)
3. [Features](#features)
4. [Prerequisites](#prerequisites)
5. [Usage](#usage)
   - [Getting Started](#getting-started)
   - [Generating Sample Data](#generating-sample-data)
   - [Using a Learner](#using-a-learner)
6. [Model Implementation Examples](#model-implementation-examples)
   - [DTLearner](#dtlearner)
   - [RTLearner](#rtlearner)
   - [LinRegLearner](#linreglearner)
   - [KNNLearner](#knnlearner)
7. [Next Step](#next-step)


## Introduction
Welcome to MLScratchPad, a hands-on repository where classic machine learning models are built from the ground up. This project is designed for those who want to understand the nitty-gritty of machine learning algorithms without relying on high-level libraries like scikit-learn. It's perfect for students, hobbyists, and professionals looking to deepen their understanding of the underlying mechanics of machine learning models.

## Models
- Classification and Regression
   - **Decision Tree Learner (`DTLearner.py`)**: Implements a basic decision tree learning algorithm.
   - **Random Tree Learner (`RTLearner.py`)**: Implements a random tree learning approach.
   - **KNN Learner (`KNNLearner.py`)**: A versatile learner for both classification and regression tasks using the K-Nearest Neighbors approach.
- Regression
-    **Linear Regression Learner (`LinRegLearner.py`)**: A foundational model for linear regression analysis.
- Classification
   - **AdaBoost Learner (`AdaBoost.py`)**: *(Note: Currently under development)* An initial implementation of the AdaBoost algorithm for boosting weaker models. This module is not yet available as an object-oriented interface.
- Ensembling
   - **Bagging Learner (`BagLearner.py`)**: Introduces the concept of bagging in machine learning.
   - **Insane Learner (`InsaneLearner.py`)**: A complex learner that combines various algorithms for enhanced learning capabilities.
- Clustering
   - **K-Means Learner (`KMeansLearner.py`)**: Implements the K-Means clustering algorithm.
   - **Spectral Learner (`SpectralLearner.py`)**: Applies spectral clustering techniques for machine learning.
   - **ISOMAP Learner (`ISOMAPLearner.py`)**: Focuses on dimensionality reduction using ISOMAP algorithm.
   - **Gaussian Mixture Model Expectation-Maximization Learner (`GMMEMLearner.py`)**: Applies GMM-EM for clustering and density estimation.
- Dimentional Reduction
   - **PCA Learner (`PCALearner.py`)**: A learner for Principal Component Analysis (PCA) for dimensionality reduction.
-Reinforcement lLearning
   - **Q-Learning Learner (`QLearner.py`)**: Demonstrates reinforcement learning using the Q-Learning algorithm.

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


### Generate Sample Data
3. (optional) Before using the learners, you can generate sample data. Use the `GenerateSampleData.py` script included in the repository to create a `sample_data.csv` file, which serves as a dataset for testing the models.

```bash
python GenerateSampleData.py
```

The script generates a dataset with various features and a target variable, saved as `sample_data.csv`. This dataset is good for trying out the models.

### Using a Learner
4. **Choose a Learner**: Decide which machine learning model you want to use. For example, if you want to use the Decision Tree Learner, you'll work with `DTLearner.py`.

5. **Prepare Your Python Environment**: Make sure you have Python installed. You can use virtual environments to manage your packages.

6. **Write Your Script**: In your Python environment, create a new Python script or open an interactive session. Import the learner class from the corresponding file. For example:
   ```python
   from DTLearner import DTLearner
   ```

7. **Initialize the Learner**: Create an instance of the learner. For instance, with `DTLearner`, you might do:
   ```python
   learner = DTLearner()
   ```

8. **Load Your Data**: Load the dataset you wish to train on. You can use any dataset in a compatible format (e.g., CSV, Excel, SQL database).

9. **Train the Learner**: Call the appropriate method to train your learner. For example:
   ```python
   learner.addEvidence(trainX, trainY)
   ```
   where `trainX` is your training input and `trainY` is the training output.

9. **Test the Learner**: Once trained, you can make predictions or evaluate the model using your test data.

Remember to consult the documentation in each learner file for specific instructions and parameters.

## Model Implementation Examples

### DTLearner
The Decision Tree Learner (DTLearner) offers a straightforward approach to decision tree modeling. Here's how to use it:
```python
# Example with Decision Tree Learner
from DTLearner import DTLearner
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Read sample data from a CSV file
df = pd.read_csv('sample_data.csv')
y = df['y'].values  # 'y' is the target column
X = df.drop('y', axis=1).values

# Manually split the data into training and testing sets
split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing
trainX, testX = X[:split_index], X[split_index:]
trainY, testY = y[:split_index], y[split_index:]

# Create a learner instance
learner = DTLearner()

# Train the learner
learner.fit(trainX, trainY)

# Make predictions
predictions = learner.query(testX)

# Calculate and print the RMSE and R²
rmse = np.sqrt(mean_squared_error(testY, predictions))
r2 = r2_score(testY, predictions)
print("RMSE:", rmse)
print("R²:", r2)
```

### RTLearner
The Random Tree Learner (RTLearner) provides an approach to machine learning based on random decision trees. Here's an example to use RTLearner with `sample_data.csv`:

```python
# Example with Random Tree Learner
from RTLearner import RTLearner
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Read sample data from a CSV file
df = pd.read_csv('sample_data.csv')
y = df['y'].values  # 'y' is the target column
X = df.drop('y', axis=1).values

# Manually split the data into training and testing sets
split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing
trainX, testX = X[:split_index], X[split_index:]
trainY, testY = y[:split_index], y[split_index:]

# Create a RTLearner instance
learner = RTLearner()

# Train the learner
learner.fit(trainX, trainY)

# Make predictions
predictions = learner.query(testX)

# Calculate and print the RMSE and R²
rmse = np.sqrt(mean_squared_error(testY, predictions))
r2 = r2_score(testY, predictions)
print("RMSE:", rmse)
print("R²:", r2)
```

(Since this model represents a single tree rather than a forest, it's common to observe lower performance compared to more complex models.)

### LinRegLearner
The Linear Regression Learner (`LinRegLearner.py`) is designed for foundational linear regression analysis. Here's how to use it:

```python
# Example with Linear Regression Learner
from LinRegLearner import LinRegLearner
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Read sample data from a CSV file
df = pd.read_csv('sample_data.csv')
y = df['y'].values  # 'y' is the target column
X = df.drop('y', axis=1).values

# Manually split the data into training and testing sets
split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing
trainX, testX = X[:split_index], X[split_index:]
trainY, testY = y[:split_index], y[split_index:]

# Create a LinRegLearner instance
learner = LinRegLearner()

# Train the learner
learner.fit(trainX, trainY)  # Use 'fit' for training

# Make predictions
predictions = learner.query(testX)

# Calculate and print the RMSE and R²
rmse = np.sqrt(mean_squared_error(testY, predictions))
r2 = r2_score(testY, predictions)
print("RMSE:", rmse)
print("R²:", r2)
```

### KNNLearner
The KNN Learner (`KNNLearner.py`) is adaptable for both classification and regression problems. It automatically detects the nature of the target variable `y` and applies the appropriate method.

```python
# Example with KNN Learner
from KNNLearner import KNNLearner
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Read sample data from a CSV file
df = pd.read_csv('sample_data.csv')
y = df['y'].values
X = df.drop('y', axis=1).values

# Manually split the data into training and testing sets
split_index = int(len(X) * 0.8)
trainX, testX = X[:split_index], X[split_index:]
trainY, testY = y[:split_index], y[split_index:]

# Create a KNNLearner instance for regression
learner = KNNLearner(k=5)

# Train the learner
learner.fit(trainX, trainY)

# Make predictions
predictions = learner.query(testX)

# Calculate and print the RMSE and R²
rmse = np.sqrt(mean_squared_error(testY, predictions))
r2 = r2_score(testY, predictions)
print("RMSE:", rmse)
print("R²:", r2)
```

## Next Step

As MLScratchPad continues to evolve, the following enhancements are planned:
- **Ensemble and Clustering Model Instructions**: Update the README.md with detailed guidelines for ensemble models and clustering models.
- **OOP Conversion**: Transform `AdaBoost.py` into object-oriented implementations.
- **Formal Testing and Visualization**: Introduce formal test datasets and code to evaluate all models, including visualizing clustering performance.
- **New Model Development**: Develop and integrate more Classification models into the repository.


