"""Persist trained models to disk."""
import os
import pickle


def save_model(estimator, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(estimator, f)
