import numpy as np
import matplotlib.pyplot as plt
import itertools

def load_data():
    X_train = np.load('X_train_regression2.npy')
    y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')
    return X_train, y_train, X_test

def scatter_plot(X_train):
    feature_combinations = list(itertools.combinations(range(X_train.shape[1]), 2))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5)

    for i, (feature_idx1, feature_idx2) in enumerate(feature_combinations):
        ax = axes[i // 3, i % 3]
        ax.scatter(X_train[:, feature_idx1], X_train[:, feature_idx2])
        ax.set_xlabel(f'Feature {feature_idx1 + 1}')
        ax.set_ylabel(f'Feature {feature_idx2 + 1}')
        ax.set_title(f'Scatter Plot: Feature {feature_idx1 + 1} vs. Feature {feature_idx2 + 1}')

    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train, X_test = load_data()
    scatter_plot(X_train) #k means or other clustering algorithm might not be the best idea
    


if __name__ == '__main__':
    main()