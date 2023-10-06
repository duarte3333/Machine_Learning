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
        #print("Name: ", name, "Error: ", error)
        if error < final_error:
            final_error = error
            final_model = model
            final_name = name     
    return final_model, final_error, final_name

def predict(model, X_train, y_train, X_test): 
    """ Predict y values for the test data using the specified model. """
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict


