import numpy as np
import matplotlib.pyplot as plt
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

def histogram(X_train):
    """ Plot a histogram of X data. """
    fig, axe = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axe = axe.flatten() #Easy way to plot the 10 subplots

    for i in range(len(X_train[0])):
        axe[i].hist(X_train[:, i],color='mediumseagreen',  bins=10) 
        axe[i].set_title('Feature')
        axe[i].set_xlabel('Value')
        axe[i].set_ylabel('Frequency')

    plt.tight_layout() #So y_label doesn't intercept other histograms
    plt.show()

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

def process_models(X_train, y_train, functions, residuals, func_model):
    if residuals is None:
        labels = func_model(X_train, y_train)
    else:
        labels = func_model(residuals, y_train)

    data_matrix = data_split(X_train, y_train, labels)
    best_model_1, best_model_2 = choose_k(functions, data_matrix[0], data_matrix[2], data_matrix[1], data_matrix[3])
    return best_model_1, best_model_2, data_matrix

def main():
    X_train, y_train, X_test = load_data()
    X_y_train = np.concatenate((X_train, y_train), axis=1)
    
    # histogram(X_train)
    # histogram(X_test) #data is normalized, so Gaussian-Mixture makes sense

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
        if action[2] == e: #We know that Gaussia-Mixture_X_y is the best model, so we save the values
            best_model_1, best_model_2, data_matrix = process_models(X_train, y_train, 
                                                                 functions, action[0], action[1])
        else:
            process_models(X_train, y_train, functions, action[0], action[1])

    y_predict_1 = np.array(predict(best_model_1, data_matrix[0], data_matrix[2], X_test))
    y_predict_2 = np.array(predict(best_model_2, data_matrix[1], data_matrix[3], X_test)).reshape(-1,1) #reshape transforms array into 2x1 array to be the same as y_predict_1
    y_predict_total = np.concatenate((y_predict_1, y_predict_2), axis=1)
    np.save('y_predicted.npy', y_predict_total)

if __name__ == '__main__':
    main()