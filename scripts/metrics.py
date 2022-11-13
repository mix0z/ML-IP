from typing import Tuple

import numpy as np
import pandas as pd


def get_tp(class_name, y_pred, y_true):
    y_class = np.full(len(y_pred), class_name)
    tp = 0
    for i, val in enumerate(y_class):
        tp += int(val == y_pred[i] == y_true[i])
    return tp


def get_fp(class_name, y_pred, y_true):
    y_class = np.full(len(y_pred), class_name)
    fp = 0
    for i, val in enumerate(y_class):
        fp += int(val == y_pred[i] != y_true[i])
    return fp


def get_fn(class_name, y_pred, y_true):
    y_class = np.full(len(y_pred), class_name)
    fn = 0
    for i, val in enumerate(y_class):
        fn += int(val == y_true[i] != y_pred[i])
    return fn


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array
                                  ) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    classes = np.unique(np.concatenate((y_pred, y_true), axis=None))
    df = pd.DataFrame()
    df['name'] = classes
    df['TP'] = df.apply(lambda row: get_tp(row['name'], y_pred, y_true), axis=1)
    df['FP'] = df.apply(lambda row: get_fp(row['name'], y_pred, y_true), axis=1)
    df['FN'] = df.apply(lambda row: get_fn(row['name'], y_pred, y_true), axis=1)
    df['recall'] = df.apply(lambda row: row['TP'] / (row['TP'] + row['FN']), axis=1)
    df['precision'] = df.apply(lambda row: row['TP'] / (row['TP'] + row['FP']), axis=1)

    accuracy = 0
    for i, val in enumerate(y_pred):
        accuracy += int(val == y_true[i])
    return df['precision'].to_numpy(), df['recall'].to_numpy(), accuracy / len(y_pred)
