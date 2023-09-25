import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.datasets as datasets
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#Carrega dados
X_train = np.load('../X_train_regression1.npy')
y_train = np.load('../y_train_regression1.npy')
#Carrega dados teste
X_test = np.load('../X_test_regression1.npy')

print(X_train.shape, X_test.shape)

# y = 1 * x_0 + 2 * x_1 + 3
model_lr = LinearRegression().fit(X_train, y_train)
y_hat = model_lr.predict(X_test)
print(model_lr.intercept_)
#reg.score(X_test, )

#reg.predict
#reg.intercept_

importance = model_lr.coef_
print(importance)
# Plot the coefficients
plt.bar(range(importance.shape[1]), importance[0])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Coefficients of the Linear Regression Model')
plt.xticks(range(importance.shape[1]), [f'Feature {i}' for i in range(importance.shape[1])])
plt.show()
