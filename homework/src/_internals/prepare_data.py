"""Data loading and train/test splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(
    url,
    *,
    target_column="quality",
    test_size=0.25,
    random_state=123456,
    sep=";",
):
    df = pd.read_csv(url, sep=sep)
    y = df[target_column]
    x = df.copy()
    x.pop(target_column)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
