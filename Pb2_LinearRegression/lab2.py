import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X_train = np.load('X_train_regression2.npy')
    y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')
    return X_train, y_train, X_test

def corr_matrix(X_train):
    correlation_matrix = np.corrcoef(X_train, rowvar=False)
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix of X_train')
    plt.xticks(range(correlation_matrix.shape[0]))
    plt.yticks(range(correlation_matrix.shape[0]))
    plt.show()

def histogram(X_train):
    plt.hist(X_train, bins=7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of training data "X"')
    plt.show()

def main():
    X_train, y_train, X_test = load_data()
    corr_matrix(X_train)
    histogram(X_train)

if __name__ == '__main__':
    main()