"""Save a model only if it improves the selected metric."""
from homework.src._internals.compare_models import is_better
from homework.src._internals.save_model import save_model


def save_model_if_better(
    estimator,
    new_metrics,
    best_metrics,
    path,
    *,
    metric="mse",
):
    if is_better(new_metrics, best_metrics, metric=metric):
        save_model(estimator, path)
        return new_metrics

    return best_metrics
