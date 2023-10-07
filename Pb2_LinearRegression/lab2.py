import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statistics import mean
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_validate

def create_lambdas(begin, end, increment):
    """ Creates a list of hyperparameter values for cross-validation. """
    return [begin + i * increment for i in range(int((end - begin) / increment))]

def linear_regression_model(X_train, y_train, k):
    """ Performs linear regression and returns the model and Mean Squared Error (MSE) using cross-validation. """
    model = linear_model.LinearRegression()
    model_scores = cross_validate(model, X_train, y_train,  # scoring='neg_mean_squared_error' to minimize the MSE
                                  cv=k, scoring='neg_mean_squared_error', return_train_score=True)
    model_MSE = abs(np.mean(model_scores['test_score']))
    return model, model_MSE, "Linear"

def ridge_regression(X_train, y_train, k):
    """ Fit ridge regression model and compute MSE using cross-validation."""
    ridge_lambdas = create_lambdas(0.1, 10, 0.01) #Define list of possible lambdas
    model = linear_model.RidgeCV(alphas=ridge_lambdas, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    ridge_lambda = model.alpha_  # Best lambda
    ridge_MSE = abs(model.best_score_)  # MSE of model
    return model, ridge_MSE, "Ridge"

def lasso_regression(X_train, y_train, k):
    """ Fit lasso regression model and compute MSE using cross-validation."""
    lasso_lambdas = create_lambdas(0.1, 10, 0.01)
    model = linear_model.LassoCV(alphas=lasso_lambdas, random_state=0,
                                 cv=k)  # CV to find best lambda, random_state= 0 to not be stochastic(avoid shuffle)
    model.fit(X_train, np.ravel(y_train)) #np.ravel is to flatten y_train
    lasso_lambda = model.alpha_  # Best lambda
    new_model = linear_model.Lasso(alpha=lasso_lambda) #Create best possible lasso model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                 scoring='neg_mean_squared_error', return_train_score=True) #Evaluate MSE
    lasso_MSE = abs(cross_model['test_score'].mean())
    return new_model, lasso_MSE, "Lasso"

def elastic_regression(X_train, y_train, k):
    """ Fit elastic net regression model and compute MSE using cross-validation."""
    elastic_ratios = create_lambdas(0.1, 0.9, 0.1) #Define possible ratio's between ridge and lasso regression 
    elastic_lambdas = create_lambdas(0.1, 10, 0.1)
    model = linear_model.ElasticNetCV(alphas=elastic_lambdas, l1_ratio=elastic_ratios, random_state=0, cv = k)
    model.fit(X_train, np.ravel(y_train))
    elastic_lambda = model.alpha_ #Best lambda
    elastic_l1_ratio = model.l1_ratio_ #Best ratio
    new_model = linear_model.ElasticNet(alpha = elastic_lambda, l1_ratio=elastic_l1_ratio) #Create best elastic model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                 scoring='neg_mean_squared_error', return_train_score=True) #Evaluate MSE
    elastic_MSE = abs(cross_model['test_score'].mean())
    return new_model, elastic_MSE, "Elastic"

def lasso_lars_regression(X_train, y_train, k): #bad idea: just for high dimensional, uncorrelated features
    """ Fit lasso lars regression model and compute MSE using cross-validation."""
    model = (linear_model.LassoLarsCV(cv=k))
    model.fit(X_train, np.ravel(y_train))
    lasso_lars_lambda = model.alpha_  # Best lambda
    new_model = linear_model.LassoLars(alpha=lasso_lars_lambda) #Best possible lasso lars model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                               scoring='neg_mean_squared_error', return_train_score=True)
    lasso_lars_MSE = abs(cross_model['test_score'].mean())
    return new_model, lasso_lars_MSE, "LassoLARS"

def bayesian_regression(X_train, y_train, k):
    """ Fit bayesian regression model and compute MSE using cross-validation. """
    model = linear_model.ARDRegression()
    model.fit(X_train, np.ravel(y_train))
    cv_results = cross_validate(model, X_train, np.ravel(y_train), cv=k, scoring='neg_mean_squared_error',
                                return_train_score=True)
    bayesian_mse = abs(cv_results['test_score'].mean())
    return model, bayesian_mse, "Bayesian"

def orthogonal_matching_pursuit_regression(X_train, y_train, k): #Good idea: problem is overdeterminated
    """ Fit OMP regression model and compute MSE using cross-validation."""
    model = (linear_model.OrthogonalMatchingPursuitCV(cv=k, max_iter=4))
    model.fit(X_train, np.ravel(y_train))
    omp_n_zero_coef = model.n_nonzero_coefs_  # Choose number of non zero coefficients that best fits the data
    new_model = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=omp_n_zero_coef, fit_intercept= True)
    cross_model = cross_validate(new_model 
        , X_train, y_train, cv=k,
        scoring='neg_mean_squared_error', return_train_score=True)
    omp_MSE = abs(cross_model['test_score'].mean())
    return new_model, omp_MSE, "OMP"

def standardization_data(X_train, X_test):
    """Standardize the data by removing the mean and scaling to unit variance."""
    scaler = preprocessing.StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)  # mean = 0 ; standard deviation = 1
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

def get_best_model(functions, X, y, k):
    """ Iterate through models and return the one with the lowest MSE."""
    final_model = None
    final_name = None
    final_error = float('inf')
    for func in functions:
        model, error, name = func(X, y, k)

        if error < final_error:
            final_error = error
            final_model = model
            final_name = name  #Save name for print   
    return final_model, final_error, final_name

def predict(model, X_train, y_train, X_test): 
    """ Predict y values for the test data using the specified model. """
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict

def load_data():
    """ Load the training and test data. """
    X_train = np.load('X_train_regression2.npy')
    y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')
    return X_train, y_train, X_test

def histogram(X_train):
    """ Plot a histogram of X data. """
    fig, axe = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axe = axe.flatten() #Easy way to plot the 4 subplots

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
    y_hat = residual_model.predict(X_train) #Predict y values to compare to y_train and obtain residuals
    residuals = (y_train - y_hat)**2
    return residuals

def gaussian__mixture_model(X_train, y_train):
    """ Perform gaussiam mixture clustering based on the residuals obtained """
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    return labels

def k_means(X_train, y_train):
    """ Perform K-means clustering based on the residuals obtained """
    #'k-means++' is a smart initialization method that spreads out the initial centroids.
    k_means = KMeans(n_clusters=2, init= 'k-means++', n_init= 10) #init is how algorithm starts, as it stands is greedy algorithm
    k_means.fit(X_train)
    cluster_indexs = k_means.fit_predict(X_train) #List that assigns binary values for index of each cluster
    return cluster_indexs

def data_split(X, y, cluster_indexs):
    """ Splits input data into two clusters based on cluster indices. """
    X_c1, X_c2 = X[cluster_indexs == 0], X[cluster_indexs == 1] #If indexs list is 0 is cluster 1, if is 1 is cluster 2
    y_c1, y_c2 = y[cluster_indexs == 0], y[cluster_indexs == 1]
    return X_c1, X_c2, y_c1, y_c2

def choose_k(functions, X_train_1, y_train_1, X_train_2, y_train_2):
    """" Iterates through possible number of folds and returns models with lowest MSE"""
    best_error = float('inf')
    best_model_1, best_model_2, best_name1, best_name2  = None, None, None, None
    for k in range(2,5):  # Number of folds, test with k-fold cross validation 
        model_1, error_1, name1 = get_best_model(functions, X_train_1, y_train_1, k)
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
    """ Calls auxiliary functions to split data and determine 2 regression models"""
    if residuals is None:
        labels = func_model(X_train, y_train)
    else:
        labels = func_model(residuals, y_train)

    data_matrix = data_split(X_train, y_train, labels)
    best_model_1, best_model_2 = choose_k(functions, data_matrix[0], data_matrix[2], data_matrix[1], data_matrix[3])
    return best_model_1, best_model_2, data_matrix

def main():
    X_train, y_train, X_test = load_data()
    X_y_train = np.concatenate((X_train, y_train), axis=1) #Concatenate X and y train for gaussian mixture clustering method
   
    histogram(X_train)
    histogram(X_test) #data is normalized, so Gaussian-Mixture makes sense

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
        if action[2] == e: #We know that Gaussia-Mixture_X_y is the best model, so we save the values for prediction
            best_model_1, best_model_2, data_matrix = process_models(X_train, y_train, 
                                                                 functions, action[0], action[1])
        else:
            process_models(X_train, y_train, functions, action[0], action[1])

    y_predict_1 = np.array(predict(best_model_1, data_matrix[0], data_matrix[2], X_test))
    y_predict_2 = np.array(predict(best_model_2, data_matrix[1], data_matrix[3], X_test)).reshape(-1,1) #reshape transforms array into 2x1 array to be the same as y_predict_1
    y_predict_total = np.concatenate((y_predict_1, y_predict_2), axis=1) #Save both predictions in a variable to save it 
    np.save('y_predicted.npy', y_predict_total)
    
if __name__ == '__main__':
    main()