import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error


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
        
def simple_approach(X_train, y_train, X_test):
    #75% of data for training and 25% for validation.
    #random state split the data the same every time you run the code.
    split_ratio = 0.5
    full_data = np.column_stack((X_train, y_train))
    X_train_split, X_validation_split, y_train_split, y_validation_split = \
        train_test_split(full_data[:, :-1], full_data[:, -1], \
            test_size=0.25, random_state=42) #random state split the data the same every time
    print("agr1", full_data[:, :-1].shape) # All lines except last feature. -> X_train
    print("arg2", full_data[:, -1].shape) # All lines of last feature. -> y_train
    model = linear_model.LinearRegression()
    model.fit(X_train_split, y_train_split)
    predictions = model.predict(X_validation_split)
    sorted_indices = np.argsort(predictions) #organize by ascendent order the predictions
    split_index = int(split_ratio * len(predictions))
    data_group1 = full_data[sorted_indices[:split_index]] #data regression1
    data_group2 = full_data[sorted_indices[split_index:]] #data regression2 
    X_group1, y_group1 = data_group1[:, :-1], data_group1[:, -1]
    X_group2, y_group2 = data_group2[:, :-1], data_group2[:, -1]
    model_group1 = linear_model.LinearRegression()
    model_group1.fit(X_group1, y_group1)
    model_group2 = linear_model.LinearRegression()
    model_group2.fit(X_group2, y_group2)
    
    # Calculate Mean Squared Error for Model 1 using validation split
    mse_model1 = mean_squared_error(y_validation_split, model_group1.predict(X_validation_split))

    # Calculate Mean Squared Error for Model 2 using validation split
    mse_model2 = mean_squared_error(y_validation_split, model_group2.predict(X_validation_split))

    print("Mean Squared Error (Model 1) on Training Data:", mse_model1)
    print("Mean Squared Error (Model 2) on Training Data:", mse_model2)
    


    
def main():
    X_train, y_train, X_test = load_data()
    print("X_train", X_train.shape)
    print("y_train",y_train.shape)
    print("X_test", X_test.shape)
    simple_approach(X_train, y_train, X_test)
    #scatter_plot(X_train) #k means or other clustering algorithm might not be the best idea
    #k_means(X_train)

if __name__ == '__main__':
    main()