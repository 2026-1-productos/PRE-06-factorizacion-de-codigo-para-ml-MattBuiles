"""Helpers to compare model metrics."""


def _metric_value(metrics, metric):
    if isinstance(metrics, dict):
        return metrics[metric]
    index_map = {"mse": 0, "mae": 1, "r2": 2}
    return metrics[index_map[metric]]


def is_better(new_metrics, best_metrics, metric="mse"):
    if best_metrics is None:
        return True

    new_value = _metric_value(new_metrics, metric)
    best_value = _metric_value(best_metrics, metric)

    if metric == "r2":
        return new_value > best_value

    return new_value < best_value
