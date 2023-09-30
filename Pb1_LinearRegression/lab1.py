import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_validate

#Function create_lambdas() takes as inputs the first and last numbers of an interval,
#as well as the increment to be added to each value inside the inverval, returning
# a list of this interval to easily define the values of hyperparameters calculated in cross validation
def create_lambdas(begin, end, increment):
    return [begin + i * increment for i in range(int((end - begin) / increment))]

#Function linear_regression_model takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
def linear_regression_model(X_train, y_train, k):
    model = linear_model.LinearRegression() #create linear regression model
    model_scores = cross_validate(model, X_train, y_train,  # scoring='neg_mean_squared_error' : minimize the MSE
                                  cv=k, scoring='neg_mean_squared_error', return_train_score=True)
    model_MSE = abs(np.mean(model_scores['test_score']))
    print("Linear - Mean squared error:", model_MSE, "\n")
    return model, model_MSE

#Function ridge_regression takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
def ridge_regression(X_train, y_train, k):
    ridge_lambdas = create_lambdas(0.1, 10, 0.01) #define list of possible lambdas
    model = linear_model.RidgeCV(alphas=ridge_lambdas, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    ridge_lambda = model.alpha_  # Best lambda
    ridge_MSE = abs(model.best_score_)  # MSE of model
    print("Ridge - Mean squared error:", ridge_MSE)
    print("Ridge - Best lambda", ridge_lambda, "\n")
    return model, ridge_MSE

#Function lasso_regression takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
def lasso_regression(X_train, y_train, k):
    lasso_lambdas = create_lambdas(0.1, 10, 0.01) #possible lambdas
    model = linear_model.LassoCV(alphas=lasso_lambdas, random_state=0,
                                 cv=k)  # CV to find best lambda, random_state= 0 to not be stochastic(avoid shuffle)
    model.fit(X_train, np.ravel(y_train)) #ravel is to flatten
    lasso_lambda = model.alpha_  # Best lambda
    new_model = linear_model.Lasso(alpha=lasso_lambda) #Create best possible lasso model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                 scoring='neg_mean_squared_error', return_train_score=True) #evaluate MSE
    lasso_MSE = abs(cross_model['test_score'].mean())
    print("Lasso - Mean squared error:", lasso_MSE)
    print("Lasso - Best lambda", lasso_lambda, "\n")
    return new_model, lasso_MSE

#Function elastic_regression takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
def elastic_regression(X_train, y_train, k):
    elastic_ratios = create_lambdas(0.1, 0.9, 0.1) #Define possible of ratio between ridge and lasso regression 
    elastic_lambdas = create_lambdas(0.1, 10, 0.1)
    model = linear_model.ElasticNetCV(alphas=elastic_lambdas, l1_ratio=elastic_ratios, random_state=0, cv = k)
    model.fit(X_train, np.ravel(y_train))
    elastic_lambda = model.alpha_ #best lambda
    elastic_l1_ratio = model.l1_ratio_ #best ratio
    new_model = linear_model.ElasticNet(alpha = elastic_lambda, l1_ratio=elastic_l1_ratio) #Create best elastic model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                 scoring='neg_mean_squared_error', return_train_score=True) #evaluate MSE
    elastic_MSE = abs(cross_model['test_score'].mean())
    print("Elastic - Mean Squared Error:", elastic_MSE)
    print("Elastic - Best ratio", elastic_l1_ratio)
    print("Elastic - Best lambda", elastic_lambda, "\n")
    return new_model, elastic_MSE

#Function lasso_lars_regression takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
def lasso_lars_regression(X_train, y_train, k): #bad idea: just for high dimensional, uncorrelated features
    model = (linear_model.LassoLarsCV(cv=k))
    model.fit(X_train, np.ravel(y_train))
    lasso_lars_lambda = model.alpha_  # Best lambda
    new_model = linear_model.LassoLars(alpha=lasso_lars_lambda) #best possible lasso lars model
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                               scoring='neg_mean_squared_error', return_train_score=True)
    lasso_lars_MSE = abs(cross_model['test_score'].mean())
    print("Lasso_lars - Mean squared error:", lasso_lars_MSE)
    print("Lasso_lars - Best lambda", lasso_lars_lambda, "\n")
    return new_model, lasso_lars_MSE

#Function bayesian_regression takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
#Due to bad results, function is not called in main()
def bayesian_regression(X_train, y_train, k):
    model = linear_model.ARDRegression()
    model.fit(X_train, np.ravel(y_train))
    cv_results = cross_validate(model, X_train, np.ravel(y_train), cv=k, scoring='neg_mean_squared_error',
                                return_train_score=True)
    bayesian_mse = abs(cv_results['test_score'].mean())
    print("bayesian -  Erro quadratico medio:", bayesian_mse, "\n")
    return model, bayesian_mse

#Function orthogonal_matching_pursuit_regression takes as inputs the training data as well as 
#the number of folds used in cross validation, returning the model and respective MSE
def orthogonal_matching_pursuit_regression(X_train, y_train, k): #good idea: problem is overdeterminated
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

#Function standardization_data takes as input X data, performs a standardization operation
#and then returns it 
def standardization_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)  # mean = 0 ; standard deviation aprox 1
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

#Function load_data() loads training and test data from file and returns it
def load_data():
    X_train = np.load('X_train_regression1.npy')
    y_train = np.load('y_train_regression1.npy')
    X_test = np.load('X_test_regression1.npy')
    return X_train, y_train, X_test

#Function corr_matrix plots a correlation matrix of X data to determine correlation between fixtures
def corr_matrix(X_train):
    correlation_matrix = np.corrcoef(X_train, rowvar=False)
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix of X_train')
    plt.xticks(range(correlation_matrix.shape[0]))
    plt.yticks(range(correlation_matrix.shape[0]))
    plt.show()

#Function histogram plots a histogram of X data
def histogram(X_train):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
    axes = axes.flatten()

    for i in range(len(X_train[0])):
        axes[i].hist(X_train[:, i], bins=5, color='blue', alpha=0.7)
        axes[i].set_title('Feature')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

#Function get_best_model iterates between all models present in this file and returns
#the one with lowest MSE
def get_best_model(functions, X, y, k):
    final_model = None
    final_error = float('inf')
    for func in functions:
        model, error = func(X, y, k)
        if error < final_error: #If MSE is lower than the last lowest, save the model
            final_error = error
            final_model = model
            
    print("final model", final_model)
    print("final error", final_error)
    return final_model

#Function predict takes as model, training data and test data and predicts values of y for test data
def predict(model, X_train, y_train, X_test): 
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict

def main():
    
    X_train, y_train, X_test = load_data() #Load data from numpy file
    X_train_norm, X_test_norm = standardization_data(X_train, X_test)  # Removing the mean and scaling to unit variance.
    histogram(X_train) #Data seems already pre-processed
    histogram(X_train_norm) #No big difference between the both histograms so we'll use X_train for lower MSE
    k =10  # Number of folds, test with k-fold cross validation 
    functions = [linear_regression_model, ridge_regression,
                 lasso_regression, lasso_lars_regression, 
                 elastic_regression, orthogonal_matching_pursuit_regression] #all models which we are testing
    best_model = get_best_model(functions, X_train, y_train, k) #choose best model
    y_predict = predict(best_model, X_train, y_train, X_test) 
    np.save('Y_Predicted.npy', y_predict) #save predicted values to file
    print(np.size(y_predict)) #print predicted values

if __name__ == '__main__':
    main()

