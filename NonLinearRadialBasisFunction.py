import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
# Carrega dados
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')
X_1d = np.asmatrix(X_train[:, 0]).T

def radial_basis_function(X_in, y):
    X = np.hstack((np.ones((X_in.shape[0], 1)), X_in))
    np.random.seed(10)
    num_basis = 8
    sigma = 0.5

    basis_idx = np.random.choice(X.shape[0], num_basis, replace=False) #random choose index basis
    #print(basis_idx)
    basis = np.array(X[basis_idx, 1]) # choose numbers that will be centroids in X_1d[:, 1]
    #print(basis)
    range = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.01) #range with steps of 0.01) ;
    #print(range)
    #for mu in  basis:
        #plt.plot(range, norm.pdf(range, mu, sigma)) #mean-mu ; standard deviation-sigma
    GBF = lambda x, mu: np.exp((-0.5)*((x - mu)/sigma)**2)
    Phi = np.asmatrix(np.ones((X[:,1].shape[0], 1))) # feita a matriz de coluna 1's, x1 e x^p com p= basis
    for mu in basis:
        col = np.array([GBF(obs, mu) for obs in np.array(X[:, 1])])
        Phi = np.column_stack((Phi, col))
    t = np.asmatrix(np.ones((range.shape[0], 1)))
    for mu in basis:
         col = np.array([GBF(obs, mu) for obs in range])
         t = np.column_stack((t, col))
    beta_hat_phi = (np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y)
    plt.plot(range, t @ beta_hat_phi, c='y', marker='.', alpha=0.5)
    plt.scatter(np.array(X[:,1]), np.array(y_train), c='b')

radial_basis_function(X_1d, y_train)
plt.show()