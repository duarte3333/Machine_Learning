import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Criar dados de exemplo (altura e peso)
altura = np.array([160, 170, 155, 180, 175]).reshape(-1, 1)  # Reshape para tornar uma coluna
peso = np.array([60, 75, 50, 85, 80]).reshape(-1, 1)  # Reshape para tornar uma coluna

# Juntar as características em uma matriz (altura e peso)
dados = np.hstack((altura, peso))

scaler2 = StandardScaler()
print(scaler2.fit(dados))
print(scaler2.mean_)
print(scaler2.transform(dados))

# Data normalization
scaler = preprocessing.StandardScaler()
scaler.fit(dados)  # Calcular a média e o desvio padrão para cada característica

# Aplicar a transformação aos dados para normalização
dados_normalizados = scaler.transform(dados)

# Exibir os dados originais e os dados normalizados
print("Dados Originais:\n", dados)
print("\nDados Normalizados:\n", dados_normalizados)
