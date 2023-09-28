import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from statistics import mean

def create_lambdas(begin, end, increment): # ]0, 10^12[ , steps of 0.1
    lambdas = [begin + i * increment for i in range(int((end - begin) / increment))]
    return lambdas

def linear_regression_model(X_train, y_train, k):
    model = linear_model.LinearRegression()
    model_scores = cross_validate(model, X_train, y_train,  # scoring='neg_mean_squared_error' : minimizar the MSE
                                  cv=k, scoring='neg_mean_squared_error', return_train_score=True)
    model_MSE = abs(np.mean(model_scores['test_score']))
    # print("linear - Erro quadrático nos dados de teste:", model_scores['test_score'])
    # print("linear - Erro quadrático nos dados de treino:", model_scores['train_score'])
    print("linear - Erro quadratico medio de teste", model_MSE, "\n")
    return model, model_MSE


def ridge_regression(X_train, y_train, k):
    ridge_lambdas = create_lambdas(0.1, 10, 0.01)
    model = linear_model.RidgeCV(alphas=ridge_lambdas, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    ridge_lambda = model.alpha_  # Melhor lambda que se ajusta
    ridge_MSE = abs(model.best_score_)  # score of the best lambda
    print("ridge - Erro quadratico medio de teste", ridge_MSE)
    print("ridge - Melhor lambda", ridge_lambda, "\n")
    return model, ridge_MSE


def lasso_regression(X_train, y_train, k):
    lasso_lambdas = create_lambdas(0.1, 10, 0.01)
    model = linear_model.LassoCV(alphas=lasso_lambdas, random_state=0,
                                 cv=k)  # CV to find best lambda, random_state= 0 para nao ser stochastic(avoid shuffle)
    model.fit(X_train, np.ravel(y_train)) #ravel is to flatten
    lasso_lambda = model.alpha_  # Melhor lambda que s   e adapta
    new_model = linear_model.Lasso(alpha=lasso_lambda)
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                 scoring='neg_mean_squared_error', return_train_score=True)
    lasso_MSE = abs(cross_model['test_score'].mean())
    print("lasso - Erro quadratico medio de teste", lasso_MSE)
    print("lasso - Melhor lambda", lasso_lambda, "\n")
    return new_model, lasso_MSE


def elastic_regression(X_train, y_train, k):
    elastic_ratios = create_lambdas(0.1, 0.9, 0.1)
    elastic_lambdas = create_lambdas(0.1, 10, 0.1)
    model = linear_model.ElasticNetCV(alphas=elastic_lambdas, l1_ratio=elastic_ratios, random_state=0, cv = k)
    model.fit(X_train, np.ravel(y_train))
    elastic_lambda = model.alpha_
    elastic_l1_ratio = model.l1_ratio_
    new_model = linear_model.ElasticNet(alpha = elastic_lambda, l1_ratio=elastic_l1_ratio)
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                 scoring='neg_mean_squared_error', return_train_score=True)
    elastic_MSE = abs(cross_model['test_score'].mean())
    print("elastic - Erro quadratico medio de teste", elastic_MSE)
    print("elastic - Melhor racio entre lambdas", elastic_l1_ratio)
    print("elactic - Melhor lambda", elastic_lambda, "\n")
    
    return new_model, elastic_MSE

def lasso_lars_regression(X_train, y_train, k): #bad idea: just for high dimensional, uncorrelated features
    model = (linear_model.LassoLarsCV(cv=k))
    model.fit(X_train, np.ravel(y_train))
    lasso_lars_lambda = model.alpha_  # Best lambda
    new_model = linear_model.LassoLars(alpha=lasso_lars_lambda)
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                               scoring='neg_mean_squared_error', return_train_score=True)
    lasso_lars_MSE = abs(cross_model['test_score'].mean())
    print("lasso_lars - Erro quadratico medio de teste", lasso_lars_MSE)
    print("lasso_lars - Melhor lambda", lasso_lars_lambda, "\n")
    return new_model, lasso_lars_MSE

def bayesian_regression(X_train, y_train, k):
    model = linear_model.ARDRegression()
    model.fit(X_train, np.ravel(y_train))
    cv_results = cross_validate(model, X_train, np.ravel(y_train), cv=k, scoring='neg_mean_squared_error',
                                return_train_score=True)
    bayesian_mse = abs(cv_results['test_score'].mean())
    print("bayesian -  Erro quadratico medio:", bayesian_mse, "\n")
    return cv_results, bayesian_mse

def orthogonal_matching_pursuit_regression(X_train, y_train, k): #good idea: problem is overdeterminated
    model = (linear_model.OrthogonalMatchingPursuitCV(cv=k, max_iter=10))
    model.fit(X_train, np.ravel(y_train))
    omp_n_zero_coef = model.n_nonzero_coefs_  # Choose number of non zero coefficients that best fits the data
    new_model = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=omp_n_zero_coef, fit_intercept= True)
    cross_model = cross_validate(new_model 
        , X_train, y_train, cv=k,
        scoring='neg_mean_squared_error', return_train_score=True)
    omp_MSE = abs(cross_model['test_score'].mean())
    print("omp - Erro quadratico medio de teste", omp_MSE)
    print("omp - Coeficiente nulos", omp_n_zero_coef, "\n")
    
    return new_model, omp_MSE

def standardization_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)  # mean = 0 ; standard deviation aprox 1
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

def load_data():
    X_train = np.load('X_train_regression1.npy')
    y_train = np.load('y_train_regression1.npy')
    X_test = np.load('X_test_regression1.npy')
    return X_train, y_train, X_test

def corr_matrix(X_train):
    correlation_matrix = np.corrcoef(X_train, rowvar=False)
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix of X_train')
    plt.xticks(range(correlation_matrix.shape[0]))
    plt.yticks(range(correlation_matrix.shape[0]))
    plt.show()

def histogram(X_train):
    plt.hist(X_train, bins=7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of training data "X"')
    plt.show()

def get_best_model(functions, X, y, k):
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
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    # print(model.coef_)
    # print(model.intercept_)
    return y_predict

def centering_data(X_train, X_test):
    X_train_cent = [i - sum(X_train)/len(X_train) for i in X_train]
    X_test_cent = [i - sum(X_test)/len(X_test) for i in X_test]
    return X_train_cent, X_test_cent

def main():
    X_train, y_train, X_test = load_data()
    X_train_norm, X_test_norm = standardization_data(X_train, X_test)  # Removing the mean and scaling to unit variance.
    # corr_matrix(X_train) #See correlation between the features of the dataset
    k = 10  # Number splits, each with 15/5=3 elements(for k=5), test with k-fold cross validation 
    functions = [linear_regression_model, ridge_regression,
                 lasso_regression, lasso_lars_regression, 
                 elastic_regression, orthogonal_matching_pursuit_regression]
    best_model = get_best_model(functions, X_train_norm, y_train, k)
    y_predict = predict(best_model, X_train_norm, y_train, X_test_norm)
    np.save('Y_Predicted.npy', y_predict)

if __name__ == '__main__':
    main()

