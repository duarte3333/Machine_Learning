import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
# Carrega dados
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')
X_1d = np.asmatrix(X_train[:, 0]).T

#np.random.seed()
#X_1d = np.random.rand(1, 20).T
#y_train = np.random.rand(1, 20).T


# print("X_train: ", X_train.shape, "y_train: ", y_train.shape, "X_train_1d: ", X_1d.shape)

def model(X_in, y, n):
    X = np.hstack((np.ones((X_in.shape[0], 1)), X_in))
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat_train = X @ beta_hat
    error_train = y - y_hat_train
    print("error1", np.mean(error_train))
    if n:
        beta_hat2 = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train)
        y_hat_train2 = X_train @ beta_hat2
        error_train2 = y - y_hat_train2
        print("error2", np.mean(error_train2))
    return beta_hat, X_in

def show_graph(X_1d, beta_hat):
    nb_points = 100
    t = np.linspace(np.min(X_1d), np.max(X_1d), nb_points)
    t = np.asmatrix(t)
    t = np.hstack((np.ones((nb_points, 1)), t.T))
    plt.scatter(np.array(X_1d), np.array(y_train), marker="o")
    plt.plot(t[:, 1], t @ beta_hat, 'xr')
    plt.title("Simple Linear Regression")


def main():
    print("First Order Model")
    beta_hat, X1d = model(X_1d, y_train, 0)
    show_graph(X_1d, beta_hat)
    plt.show()
    # print("General Case")
    # model(X_train, y_train, 1)

main()
from NonLinearRadialBasisFunction import *
from PolynomialRegression import*

model (ahsdgaj)