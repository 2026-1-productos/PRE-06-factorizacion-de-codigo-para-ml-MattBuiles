#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

import os
import pickle

# importacion de librerias
from sklearn.neighbors import KNeighborsRegressor

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics

# descarga de datos y division
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
(x_train, x_test, y_train, y_test) = prepare_data(url)

# entrenar el modelo
estimator = KNeighborsRegressor(n_neighbors=5)
estimator.fit(x_train, y_train)

# Guardar el estimador entrenado
os.makedirs("models", exist_ok=True)
with open(os.path.join("models", "knn.pkl"), "wb") as f:
    pickle.dump(estimator, f)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
y_pred = estimator.predict(x_train)
mse, mae, r2 = calculate_metrics(y_train, y_pred)

print()
print("Metricas de entrenamiento:")
print_metrics(mse, mae, r2)

# Metricas de error durante testing
print()
print("Metricas de testing:")
y_pred = estimator.predict(x_test)
mse, mae, r2 = calculate_metrics(y_test, y_pred)

print_metrics(mse, mae, r2)
