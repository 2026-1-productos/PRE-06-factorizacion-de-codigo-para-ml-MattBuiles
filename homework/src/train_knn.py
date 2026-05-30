#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from sklearn.neighbors import KNeighborsRegressor
from ._internals.run_experiment import run_experiment


run_experiment(
    estimator=KNeighborsRegressor(
        n_neighbors=5,
    )
)
