#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# importacion de librerias
from sklearn.linear_model import ElasticNet
from ._internals.run_experiment import run_experiment

run_experiment(
    estimator=ElasticNet(
        alpha=0.5,
        l1_ratio=0.5,
        random_state=12345,
    )
)
