import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

""" Dermoscopy Image - Imagem de Dermatoscopia:
-Melanoma: Type of skin cancer that produces a pigment in the skin 
-Nervus: Benign growth on the skin, non-cancerous and harmless """

def load_data():
    """ Load the training and test data. """
    #2352 elements each image: 2352 = pixels x pixels x 3 (RGB)
    X_train = np.load('Xtrain_Classification1.npy') #6254 x 2352. 2352 = pixels x pixels x 3 (RGB). Training set is inbalanced, different number of samples for each class
    y_train = np.load('ytrain_Classification1.npy') #6254. 1D vector
    X_test = np.load('Xtest_Classification1.npy') #1764 x 2352. Has data from two distinct sources
    return X_train, y_train, X_test

def standardization(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) #apply same transformation to test data
    return X_train, X_test

def create_lambdas(begin, end, increment):
    """ Creates a list of hyperparameter values for cross-validation. """
    return [begin + i * increment for i in range(int((end - begin) / increment))]

def MLP(X_train, y_train, parameters):
    classifier = MLPClassifier(solver= 'adam') #Keep default solver 'adam' since is better for larger datasets and is robust
    grid_search = GridSearchCV(classifier, parameters) 
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    print(best_estimator)
    
def show_images(X_train, y_train):
    num_images_to_display = 30
    num_images_per_row = 5
    num_rows = num_images_to_display // num_images_per_row

    plt.figure(figsize=(15, 3 * num_rows))

    for i in range(num_images_to_display):
        plt.subplot(num_rows, num_images_per_row, i + 1)
        plt.imshow(X_train[i].reshape(28, 28, 3))  # Assuming images are 28x28 pixels with 3 channels (RGB)
        plt.title(f"Class {y_train[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()    

def main():
    X_train, y_train, X_test = load_data()
    show_images(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    MLP(X_train, y_train, parameters)
    #X_train, X_test = standardization(X_train, X_test)
    

if __name__ == '__main__':
    main()
    1