import numpy as np
import matplotlib.pyplot as plt
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
    print("Linear - Mean squared error:", model_MSE, "\n")
    return model, model_MSE

def ridge_regression(X_train, y_train, k):
    """ Fit ridge regression model and compute MSE using cross-validation."""
    ridge_lambdas = create_lambdas(0.1, 10, 0.01) #Define list of possible lambdas
    model = linear_model.RidgeCV(alphas=ridge_lambdas, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    ridge_lambda = model.alpha_  # Best lambda
    ridge_MSE = abs(model.best_score_)  # MSE of model
    print("Ridge - Mean squared error:", ridge_MSE)
    print("Ridge - Best lambda", ridge_lambda, "\n")
    return model, ridge_MSE

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
    print("Lasso - Mean squared error:", lasso_MSE)
    print("Lasso - Best lambda", lasso_lambda, "\n")
    return new_model, lasso_MSE

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
    print("Elastic - Mean Squared Error:", elastic_MSE)
    print("Elastic - Best ratio", elastic_l1_ratio)
    print("Elastic - Best lambda", elastic_lambda, "\n")
    return new_model, elastic_MSE

def lasso_lars_regression(X_train, y_train, k): #bad idea: just for high dimensional, uncorrelated features
    """ Fit lasso lars regression model and compute MSE using cross-validation."""
    model = (linear_model.LassoLarsCV(cv=k))
    model.fit(X_train, np.ravel(y_train))
    lasso_lars_lambda = model.alpha_  # Best lambda
    new_model = linear_model.LassoLars(alpha=lasso_lars_lambda) #Best possible lasso lars model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                               scoring='neg_mean_squared_error', return_train_score=True)
    lasso_lars_MSE = abs(cross_model['test_score'].mean())
    print("Lasso_lars - Mean squared error:", lasso_lars_MSE)
    print("Lasso_lars - Best lambda", lasso_lars_lambda, "\n")
    return new_model, lasso_lars_MSE

def bayesian_regression(X_train, y_train, k):
    """ Fit bayesian regression model and compute MSE using cross-validation. """
    model = linear_model.ARDRegression()
    model.fit(X_train, np.ravel(y_train))
    cv_results = cross_validate(model, X_train, np.ravel(y_train), cv=k, scoring='neg_mean_squared_error',
                                return_train_score=True)
    bayesian_mse = abs(cv_results['test_score'].mean())
    print("Bayesian -  Mean squared: error", bayesian_mse, "\n")
    return model, bayesian_mse

def orthogonal_matching_pursuit_regression(X_train, y_train, k): #Good idea: problem is overdeterminated
    """ Fit OMP regression model and compute MSE using cross-validation."""
    model = (linear_model.OrthogonalMatchingPursuitCV(cv=k, max_iter=10))
    model.fit(X_train, np.ravel(y_train))
    omp_n_zero_coef = model.n_nonzero_coefs_  # Choose number of non zero coefficients that best fits the data
    new_model = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=omp_n_zero_coef, fit_intercept= True)
    cross_model = cross_validate(new_model 
        , X_train, y_train, cv=k,
        scoring='neg_mean_squared_error', return_train_score=True)
    omp_MSE = abs(cross_model['test_score'].mean())
    print("Omp - Mean Squared Error", omp_MSE)
    print("Omp - Number of zero coefficients", omp_n_zero_coef, "\n")
    return new_model, omp_MSE

def standardization_data(X_train, X_test):
    """Standardize the data by removing the mean and scaling to unit variance."""
    scaler = preprocessing.StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)  # mean = 0 ; standard deviation = 1
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

def load_data():
    """ Load the training and test data. """
    X_train = np.load('X_train_regression1.npy')
    y_train = np.load('y_train_regression1.npy')
    X_test = np.load('X_test_regression1.npy')
    return X_train, y_train, X_test

def corr_matrix(X_train):
    """ Plot a correlation matrix of X data to determine correlation between features. """
    correlation_matrix = np.corrcoef(X_train, rowvar=False)
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix of X_train')
    plt.xticks(range(correlation_matrix.shape[0]))
    plt.yticks(range(correlation_matrix.shape[0]))
    plt.show()

def histogram(X_train):
    """ Plot a histogram of X data. """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
    axes = axes.flatten()

    for i in range(len(X_train[0])):
        axes[i].hist(X_train[:, i], bins=5, color='blue', alpha=0.7)
        axes[i].set_title('Feature')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def get_best_model(functions, X, y, k):
    """ Iterate through models and return the one with the lowest MSE."""
    final_model = None
    final_error = float('inf')
    for func in functions:
        model, error = func(X, y, k)
        if error < final_error:
            final_error = error
            final_model = model      
    print("final model", final_model)
    print("final error", final_error)
    return final_model

def predict(model, X_train, y_train, X_test): 
    """ Predict y values for the test data using the specified model. """
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict

def main():
    """ Load data, preprocess, select best model, and make predictions. """
    X_train, y_train, X_test = load_data()
    X_train_norm, X_test_norm = standardization_data(X_train, X_test) 
    histogram(X_train) #Data seems already pre-processed
    histogram(X_train_norm) #No big difference between the both histograms so we'll use X_train for lower MSE
    k =10  # Number of folds, test with k-fold cross validation 
    functions = [linear_regression_model, ridge_regression,
                 lasso_regression, lasso_lars_regression, 
                 bayesian_regression,
                 elastic_regression, orthogonal_matching_pursuit_regression] #All models which we are testing
    best_model = get_best_model(functions, X_train, y_train, k)
    y_predict = predict(best_model, X_train, y_train, X_test) 
    np.save('Y_Predicted.npy', y_predict)

if __name__ == '__main__':
    main()

