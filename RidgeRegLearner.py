######################################################################
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#####################################################################

def ridge_regression(X_train, y_train, X_test, y_test, lamb, sigma):
    n_train = len(X_train)

    # Design matrix for training data
    X_train = np.column_stack((np.ones(n_train), X_train))

    # Design matrix for test data
    X_test = np.array([1, X_test])

    # Ridge regression solution
    lambda_I = lamb * np.eye(2)  # Regularization term
    beta_hat = np.linalg.inv(X_train.T @ X_train + lambda_I) @ X_train.T @ y_train

    # Prediction on the test sample
    y_pred_test = X_test @ beta_hat

    # Calculate MSE as Bias^2 + Variance
    bias_squared = (y_test - np.mean(y_pred_test)) ** 2
    variance = sigma*np.var(X_train @ beta_hat)
    mse = bias_squared + variance

    return bias_squared, variance, mse

if __name__ == '__main__':
    # part (e)
    beta_0 = -1
    beta_1 = 1
    sigma = 1
    X_train = np.array([0.1, 1])
    y_train = beta_0 + beta_1 * X_train
    X_test = 0.75
    y_test = beta_0 + beta_1 * X_test

    lamb_array = np.arange(0,10,0.2)
    B2 = np.ones(len(lamb_array))
    V = np.ones(len(lamb_array))
    M = np.ones(len(lamb_array))
    for i in range(len(lamb_array)):
        lamb = lamb_array[i]
        B2[i], V[i], M[i] = ridge_regression(X_train=X_train,
                                                       y_train=y_train,
                                                       X_test=X_test,
                                                       y_test=y_test,
                                                       lamb=lamb,
                                                       sigma=sigma)

    plt.title('Error of Ridge Regression(x = 0.75)')
    plt.xlabel('Regularization Parameter Lambda (x = 0.75)')
    plt.ylabel('Error')
    plt.grid(True, zorder=-10)
    plt.plot(lamb_array, B2, 'b', label='Squared Bias', marker='.')
    plt.plot(lamb_array, V, 'g', label='Variance', marker='.')
    plt.plot(lamb_array, M, 'r', label='Mean squared error(MSE)', marker='.')
    plt.legend()
    plt.savefig('RR1.png')
    plt.close()

    # part (f)
    X_test = 0.01
    y_test = beta_0 + beta_1 * X_test

    for i in range(len(lamb_array)):
        lamb = lamb_array[i]
        B2[i], V[i], M[i] = ridge_regression(X_train=X_train,
                                                       y_train=y_train,
                                                       X_test=X_test,
                                                       y_test=y_test,
                                                       lamb=lamb,
                                                       sigma=sigma)

    plt.title('Error of Ridge Regression(x = 0.01)')
    plt.xlabel('Regularization Parameter Lambda (x = 0.01)')
    plt.ylabel('Error')
    plt.grid(True, zorder=-10)
    plt.plot(lamb_array, B2, 'b', label='Squared Bias', marker='.')
    plt.plot(lamb_array, V, 'g', label='Variance', marker='.')
    plt.plot(lamb_array, M, 'r', label='Mean squared error(MSE)', marker='.')
    plt.legend()
    plt.savefig('RR2.png')
    plt.close()











