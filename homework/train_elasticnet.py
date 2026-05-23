#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

import os
import pickle

# importacion de librerias
from sklearn.linear_model import ElasticNet

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics

# descarga de datos y division
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
(x_train, x_test, y_train, y_test) = prepare_data(url)

# entrenar el modelo
estimator = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=12345)
estimator.fit(x_train, y_train)

# Guardar el estimador entrenado
os.makedirs("models", exist_ok=True)
with open(os.path.join("models", "estimator.pkl"), "wb") as f:
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
