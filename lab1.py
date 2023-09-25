import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from statistics import mean

def linear_regression_model(X_train, y_train, k):
    model = linear_model.LinearRegression()
    model_scores = cross_validate(model, X_train, y_train, #scoring='neg_mean_squared_error' : minimizar the MSE
                                    cv=k, scoring='neg_mean_squared_error', return_train_score=True)
    model_MSE = abs(np.mean(model_scores['test_score']))
    print("linear - Erro quadrático nos dados de teste:", model_scores['test_score'])
    print("linear - Erro quadrático nos dados de treino:", model_scores['train_score'])
    print("linear - Erro quadratico medio de teste", model_MSE, "\n")
    return model

def ridge_regression(X_train, y_train, k):
    increment = 5 / 200
    ridge_lambdas = [i * increment for i in range(1, 200)] #200 val de ]0,5[
    model = linear_model.RidgeCV(alphas=ridge_lambdas, scoring='neg_mean_squared_error', cv=k)
    model.fit(X_train, y_train)
    ridge_lambda = model.alpha_  # Melhor lambda que se ajusta
    ridge_MSE = abs(model.best_score_)  # score of the best lambda
    print("ridge - Erro quadratico medio de teste", ridge_MSE)
    print("ridge - Melhor lambda", ridge_lambda, "\n")

def lasso_regression(X_train, y_train, k):
    increment = 5 / 200
    lasso_lambdas = [i * increment for i in range(1, 200)] #200 val de ]0,5[
    model = linear_model.LassoCV(alphas=lasso_lambdas, random_state=0, cv=k) #CV to find best lambda, random_state= 0 para nao ser stochastic(avoid shuffle)
    model.fit(X_train, np.ravel(y_train))
    lasso_lambda = model.alpha_  #Melhor lambda que se adapta
    new_model = linear_model.Lasso(alpha=lasso_lambda)
    cross_model = cross_validate(new_model, X_train, y_train, cv=k,
                                      scoring='neg_mean_squared_error', return_train_score=True)
    lasso_MSE = abs(cross_model['test_score'].mean())
    print("lasso - Erro quadratico medio de teste", lasso_MSE)
    print("lasso - Melhor lambda", lasso_lambda, "\n")

def normalize_data(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)  # mean = 0 ; standard deviation aprox 1
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

def predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    #print(model.coef_)
    #print(model.intercept_)
    return y_predict

def load_data():
    X_train = np.load('X_train_regression1.npy')
    y_train = np.load('y_train_regression1.npy')
    X_test = np.load('X_test_regression1.npy')
    return X_train, y_train, X_test

def main():
    X_train, y_train, X_test = load_data()
    X_train_norm, X_test_norm = normalize_data(X_train, X_test) # Removing the mean and scaling to unit variance.
    k = 10  # Number splits, each with 15/5=3 elements(for k=5), test with k-fold cross validation
    model = linear_regression_model(X_train_norm, y_train, k)
    ridge_regression(X_train_norm, y_train, k)
    lasso_regression(X_train_norm, y_train, k)
    y_predict = predict(model, X_train_norm, y_train, X_test_norm)

main()
