"""Functions for data preprocessing."""
import os
from typing import Tuple

import numpy as np
import pandas as pd


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Read cancer dataset.

    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).


    """
    df = pd.read_csv(path_to_csv).sample(frac=1).reset_index(drop=True)
    return (
        df.drop(columns="label").to_numpy(),
        df["label"].apply(lambda row: 1 if row == "B" else 0).to_numpy(),
    )


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Read spam dataset.

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    df = pd.read_csv(path_to_csv).sample(frac=1).reset_index(drop=True)
    return df.drop(columns="label").to_numpy(), df["label"].to_numpy()


def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Split dataset into train and test parts.

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    df = pd.DataFrame(X)
    df["label"] = y

    train_size = int(ratio * len(df.index))
    test_size = len(df.index) - train_size

    df_train, df_test = df.head(train_size), df.tail(test_size)

    return (
        df_train.drop(columns="label").to_numpy(),
        df_train["label"].to_numpy(),
        df_test.drop(columns="label").to_numpy(),
        df_test["label"].to_numpy(),
    )


if __name__ == "__main__":
    X, y = read_cancer_dataset("cancer.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    if not os.path.exists("data"):
        os.mkdir("data")
    X_train.dump("data/X_train.npy")
    y_train.dump("data/y_train.npy")
    X_test.dump("data/X_test.npy")
    y_test.dump("data/y_test.npy")
