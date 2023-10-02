import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.cluster import KMeans

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

def heatmap(X_train):
    """ If we go the route of doing hierarchical clustering. """

    plt.figure(figsize=(8, 6))  
    sns.heatmap(X_train, cmap="RdBu_r")  
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    plt.title("Heatmap of X_train")
    plt.show()

def k_means(X_train):
    k_means = KMeans(n_clusters=2, init= 'k-means++', n_init= 10) #init is how algorithm stars, as it stands is greedy algorithm
    k_means.fit(X_train)
    cluster_labels = k_means.fit_predict(X_train)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=cluster_labels, cmap='rainbow')  # Adjust the features as needed

    # Customize the plot with labels and a title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering Results')

    # Optionally, you can show the cluster centers as well
    cluster_centers = k_means.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label='Centroids')

    plt.legend()
    plt.show()
        
def main():
    X_train, y_train, X_test = load_data()
    #scatter_plot(X_train) #k means or other clustering algorithm might not be the best idea
    k_means(X_train)

if __name__ == '__main__':
    main()