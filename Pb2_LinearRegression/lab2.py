import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn import linear_model
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

# def heatmap(X_train):
#     """ If we go the route of doing hierarchical clustering. """

#     plt.figure(figsize=(8, 6))  
#     sns.heatmap(X_train, cmap="RdBu_r")  
#     plt.xlabel("Feature")
#     plt.ylabel("Sample")
#     plt.title("Heatmap of X_train")
#     plt.show()

def residual_func(X_train, y_train):
    residual_model = linear_model.LinearRegression()
    residual_model.fit(X_train, y_train)
    y_hat = residual_model.predict(X_train)
    residuals = y_train - y_hat

    return residuals

def k_means(X_train, y_train):
    k_means = KMeans(n_clusters=2, init= 'k-means++', n_init= 10) #init is how algorithm stars, as it stands is greedy algorithm
    k_means.fit(X_train)
    cluster_indexs = k_means.fit_predict(X_train)
    cluster_centers = k_means.cluster_centers_
   
    return cluster_centers, cluster_indexs

def data_split(X, y, cluster_indexs):
    
    X_1 = X[cluster_indexs == 0] #Only values where index is 0 (1st cluster)
    X_2 = X[cluster_indexs == 1] #Only values where index is 1 (2nd cluster)
    y_1 = y[cluster_indexs == 0]
    y_2 = y[cluster_indexs == 1]

    return X_1, X_2, y_1, y_2

def choose_k(functions, X_train_1, y_train_1, X_train_2, y_train_2):
    error_k = float('inf')
    for k in range(2,5):  # Number of folds, test with k-fold cross validation 
        functions = [linear_regression_model, ridge_regression,
                    lasso_regression, lasso_lars_regression, 
                    bayesian_regression,
                    elastic_regression]
        best_model_1, final_error_1 = get_best_model(functions, X_train_1, y_train_1, k)
        best_model_2, final_error_2 = get_best_model(functions, X_train_2, y_train_2, k)

        if mean([final_error_1, final_error_2]) < error_k:
            error_k = mean([final_error_1, final_error_2])
            k_final = k
    print('The number of folds that minimizes MSE is k =',k,'resulting in a MSE of', error_k)
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
    residuals = residual_func(X_train, y_train)
    cluster_centers, cluster_indexs = k_means(residuals, y_train)
    X_train_1, X_train_2, y_train_1, y_train_2 = data_split(X_train, y_train, cluster_indexs)
    #scatter_plot(X_train) #k means or other clustering algorithm might not be the best idea
    #k_means_plot(X_train, cluster_indexs, cluster_centers)
    functions = [linear_regression_model, ridge_regression,
                 lasso_regression, lasso_lars_regression, 
                 bayesian_regression,
                 elastic_regression] #orthogonal_matching_pursuit_regression is disabled as it stands
    best_model_1, best_model_2 = choose_k(functions, X_train_1, y_train_1, X_train_2, y_train_2)

if __name__ == '__main__':
    main()