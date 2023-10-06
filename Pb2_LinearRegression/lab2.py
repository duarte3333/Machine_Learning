import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statistics import mean
from regression_models import *

def load_data():
    """ Load the training and test data. """
    X_train = np.load('X_train_regression2.npy')
    y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')
    return X_train, y_train, X_test

def residual_func(X_train, y_train, functions):
    """ Calculate residuals using a linear regression model. """
    residual_model, final_error, name = get_best_model(functions, X_train, y_train, 10)
    print("Residual Model choosen: ", name)
    print("Final error", final_error)
    residual_model.fit(X_train, y_train)
    y_hat = residual_model.predict(X_train)
    residuals = (y_train - y_hat)

    return residuals

def gaussian__mixture_model(X_train, y_train):
    """ Perform K-means clustering based on the residuals obtained """
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    return labels

def k_means(X_train, y_train):
    """ Perform K-means clustering based on the residuals obtained """
    #'k-means++' is a smart initialization method that spreads out the initial centroids.
    k_means = KMeans(n_clusters=2, init= 'k-means++', n_init= 10) #init is how algorithm starts, as it stands is greedy algorithm
    k_means.fit(X_train)
    cluster_indexs = k_means.fit_predict(X_train)
    cluster_centers = k_means.cluster_centers_
    return cluster_indexs


def data_split(X, y, cluster_indexs):
    """ Splits input data into two clusters based on cluster indices. """
    X_c1, X_c2 = X[cluster_indexs == 0], X[cluster_indexs == 1]
    y_c1, y_c2 = y[cluster_indexs == 0], y[cluster_indexs == 1]
    return X_c1, X_c2, y_c1, y_c2
    

def choose_k(functions, X_train_1, y_train_1, X_train_2, y_train_2):
    best_error = float('inf')
    best_model_1, best_model_2, best_name1, best_name2  = None, None, None, None
    for k in range(2,5):  # Number of folds, test with k-fold cross validation 
        #print(">Current value of K: ", k)
        #print(">>Model 1")
        model_1, error_1, name1 = get_best_model(functions, X_train_1, y_train_1, k)
        #print(">>Model 2")
        model_2, error_2, name2 = get_best_model(functions, X_train_2, y_train_2, k)
        mean_error = mean([error_1, error_2])
        if mean_error < best_error:
            best_error = mean_error
            final_error_1, final_error_2 = error_1,  error_2
            best_model_1, best_model_2  = model_1, model_2
            best_name1, best_name2 = name1, name2
            k_final = k
    print('The number of folds that minimizes MSE is k =',k_final,'resulting in a MSE of', best_error)
    print("Best model 1: ", best_name1, " Error model 1: ", final_error_1)    
    print("Best model 2: ", best_name2, " Error model 2: ", final_error_2)    
    return best_model_1, best_model_2

def k_means_plot(X_train, cluster_indexs, cluster_centers):
    #Plot all 6 combinations of cluters
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5)

    combinations = list(itertools.combinations(range(X_train.shape[1]), 2))
    for i, (feature_idx1, feature_idx2) in enumerate(combinations):
        ax = axes[i // 3, i % 3]
        scatter = ax.scatter(X_train[:, feature_idx1], X_train[:, feature_idx2], c=cluster_indexs, cmap='rainbow')
        ax.set_xlabel(f'Feature {feature_idx1 + 1}')
        ax.set_ylabel(f'Feature {feature_idx2 + 1}')
        ax.set_title(f'Combination: {feature_idx1 + 1}, {feature_idx2 + 1}')

    # Optionally, you can show the cluster centers as well
    
    for i, (feature_idx1, feature_idx2) in enumerate(combinations):
        ax = axes[i // 3, i % 3]
        ax.scatter(cluster_centers[:, feature_idx1], cluster_centers[:, feature_idx2], c='black', marker='x', s=100, label='Centroids')

    plt.tight_layout()
    plt.show()
    
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

def process_models(X_train, y_train, functions, residuals, func_model):
    if residuals is None:
        labels = func_model(X_train, y_train)
    else:
        labels = func_model(residuals, y_train)

    X_train_1, X_train_2, y_train_1, y_train_2 = data_split(X_train, y_train, labels)
    best_model_1, best_model_2 = choose_k(functions, X_train_1, y_train_1, X_train_2, y_train_2)
    return best_model_1, best_model_2

def main():
    X_train, y_train, X_test = load_data()
    
    X_y_train = np.concatenate((X_train, y_train), axis=1)
    print(X_y_train.shape)
    
    functions = [linear_regression_model, ridge_regression,
                 lasso_regression, lasso_lars_regression, 
                 bayesian_regression,
                 elastic_regression, orthogonal_matching_pursuit_regression]
    
    print("-------------------------RESIDUALS-----------------------------")
    residuals = residual_func(X_train, y_train, functions)
    a = "-------------------------K-MEANS-------------------------------"
    b = "----------------------GAUSSIAN-MIXTURE-------------------------"
    c = "----------------K-MEANS WITHOUT RESIDUALS----------------------"
    d = "----------------GAUSSIAN-MIXTURE WITHOUT RESIDUALS-------------"
    e = "--------------------GAUSSIAN-MIXTURE_X_Y-----------------------"
    f = "------------------------K-MEANS_X_Y----------------------------"
    
    actions = [(residuals, k_means, a), 
               (residuals, gaussian__mixture_model, b),
               (None, k_means, c), 
               (None, gaussian__mixture_model, d),
               (X_y_train, gaussian__mixture_model, e), 
               (X_y_train, k_means, f)]
    
    for action in actions:
        print(action[2])
        process_models(X_train, y_train, functions, action[0], action[1])
        
   
    #scatter_plot(X_train) #k means or other clustering algorithm might not be the best idea
    #k_means_plot(X_train, cluster_indexs, cluster_centers)

if __name__ == '__main__':
    main()