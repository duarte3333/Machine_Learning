import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
# Carrega dados
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')
X_1d = np.asmatrix(X_train[:, 0]).T

def normalize(m):
    for j in range(1, np.shape(m)[1]):
        m[:, j] = (m[:, j] - np.mean(m[:, j])) / np.std(m[:, j])
    return m

def polynomial_model(X_in, y):
    power = 10
    Xp = np.hstack((np.ones((X_in.shape[0], 1)), X_in))  # Include a column of ones
    new = [np.power(np.array(Xp[:, 1]), i) for i in range(power)]
    phiP = np.asmatrix(np.column_stack(new))
    phiP = normalize(phiP)
    print("phiP", phiP.shape)
    beta_hat_phiP = np.linalg.inv(phiP.T @ phiP) @ phiP.T @ y
    print("beta", beta_hat_phiP.shape)
    return beta_hat_phiP, phiP, Xp

def show_graph_polynomial(phiP, beta_hat_phiP, Xp, y):
    nb_points = 100
    t = np.linspace(np.min(phiP[:, 1]), np.max(phiP[:, 1]), nb_points)
    Phi_t = np.asmatrix(np.column_stack([np.array(t)**i for i in range(10)]))
    Phi_t = normalize(Phi_t)
    print("phi_T", Phi_t.shape)
    print("t", t.shape)

    #plt.plot(Xp[:, 1], y, '.')
    plt.scatter(np.array(Xp[:, 1]), np.array(y_train), c='b')
    plt.plot(t, Phi_t @ beta_hat_phiP, 'xy')

# Generate some example data (replace with your actual data)
#X_1d = np.random.random((100, 1))
#y_train = np.random.random((100, 1))
#print(X_1d.shape)
#print(y_train.shape)
beta_hat_phiP, phiP, Xp = polynomial_model(X_1d, y_train)
show_graph_polynomial(phiP, beta_hat_phiP, Xp, y_train)
plt.show()
