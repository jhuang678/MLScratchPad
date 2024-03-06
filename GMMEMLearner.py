import numpy as np
import scipy.io

import PCALearner as pca
from scipy.stats import multivariate_normal as mvn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class GMMEMLearner(object):
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.log_likelihoods = []  # List to store log-likelihood values
        if self.verbose:
            print("\nInitialized GMM-EM Learner")

    def author(self):
        return "jhuang678"

    def initialization(self, m, n, k):
        pi = np.random.random(k)
        pi = pi / np.sum(pi)
        print("pi")
        print(pi.shape)
        print(pi)
        print()

        mu = np.random.normal(0, 1, size=(k, n))
        print("mu")
        print(mu.shape)
        print(mu)
        print()

        sigma = []
        for _ in range(k):
            S = np.random.normal(0, 1, size=(n, n))
            sigma.append(S @ S.T + np.eye(n))
        print("sigma")
        print(len(sigma))
        print(sigma[0].shape)
        print(sigma[0])
        print()

        tau = np.full((m, k), fill_value=0.)
        print("tau")
        print(tau.shape)
        print(tau)
        print()
        return pi, mu, sigma, tau

    def e_step(self, X, pi, mu, sigma, tau):
        k = len(pi)
        log_likelihood = 0.0  # Initialize the log likelihood
        for ki in range(k):
            tau[:, ki] = pi[ki] * mvn.pdf(X, mu[ki], sigma[ki])
            log_likelihood += np.log(np.sum(tau[:, ki]))
        # Normalize tau and get likelihood
        sum_tau = np.sum(tau, axis=1)
        sum_tau.shape = (X.shape[0], 1)
        tau = np.divide(tau, np.tile(sum_tau, (1, k)))
        return tau, log_likelihood

    def m_step(self, X, pi, mu, sigma, tau):
        k = len(pi)
        m = X.shape[0]
        for ki in range(k):
            # Update prior
            pi[ki] = np.sum(tau[:, ki])/m

            # Update component mean
            mu[ki] = X.T @ tau[:, ki] / np.sum(tau[:, ki], axis=0)

            # update cov matrix
            dummy = X - np.tile(mu[ki], (m, 1))  # X-mu
            sigma[ki] = dummy.T @ np.diag(tau[:, ki]) @ dummy / np.sum(tau[:, ki], axis=0)
        return pi, mu, sigma

    def train_data(self, X, k=2, max_iter=100, tol=1e-5, plot=True):
        # (1) Initialization
        pi, mu, sigma, tau = self.initialization(m=X.shape[0], n=X.shape[1], k=k)
        mu_original = mu.copy()

        for i in range(max_iter):
            # (2) E-Step
            tau, log_likelihood = self.e_step(X, pi, mu, sigma, tau)
            # (2) M-Step
            pi, mu, sigma = self.m_step(X, pi, mu, sigma, tau)

            print('-----iteration---', i)

            print("Log Likelihood:", log_likelihood)
            self.log_likelihoods.append(log_likelihood)  # Append log-likelihood value to the list

            if np.linalg.norm(mu - mu_original) < tol:
                print('Training converged.')
                break
            mu_original = mu.copy()
            if i == max_iter:
                print('Max iteration reached.')
                break
        # Plot the line chart
        if plot == True:
            plot_fig = plt.figure()
            plt.plot(range(1,len(self.log_likelihoods)), self.log_likelihoods[1:])
            plt.xlabel('Iterations')
            plt.ylabel('Log-Likelihood')
            plt.title('EM-Algorithm: Log-Likelihood V.S. Iteration')
            plot_fig.savefig('output/log_likelihood.png')

        return pi, mu, sigma, tau

if __name__ == "__main__":
    print("This is a Gaussian Mixture Model and EM Learner.")



