import numpy as np

def load_data():
    """ Load the training and test data. """
    X_train = np.load('Xtrain_Classification1.npy') #6254 x 2352. 2352 = pixels x pixels x 3 (RGB). Training set is inbalanced, different number of samples for each class
    y_train = np.load('ytrain_Classification1.npy') #6254. 1D vector
    X_test = np.load('Xtest_Classification1.npy') #1764 x 2352. Has data from two distinct sources
    return X_train, y_train, X_test


def main():
    X_train, y_train, X_test = load_data()
    

if __name__ == '__main__':
    main()