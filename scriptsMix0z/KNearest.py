"""Implementation of K-Nearest Neighbors algorithm."""
from typing import List, NoReturn

import numpy as np
from KDTree import KDTree


class KNearest:
    """K Nearest Neighbors classifier."""

    tree = None
    n_neighbors = None
    leaf_size = None
    y_train = None
    classes = None
    class_to_indexes = None

    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """
        Initialize K Nearest Neighbors classifier.

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(X, leaf_size=self.leaf_size)
        self.y_train = y
        self.classes = sorted(np.unique(y))
        self.class_to_indexes = dict()
        for clas in self.classes:
            self.class_to_indexes[clas] = []
            for i in range(0, len(y)):
                if y[i] == clas:
                    self.class_to_indexes[clas].append(i)

    def predict_proba(self, X: np.array) -> List[np.array]:
        """
        Get probabilities for each class.

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """
        ans_arr = []
        predicted = self.tree.query(X, k=self.n_neighbors)
        for ind, _ in enumerate(X):
            tmp_arr = []
            for clas in self.classes:
                summ = 0
                for i in predicted[ind]:
                    if i in self.class_to_indexes[clas]:
                        summ += 1
                tmp_arr.append(summ / len(predicted))
            ans_arr.append(np.array(tmp_arr))

        return ans_arr

    def predict(self, X: np.array) -> np.array:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.

        """
        return np.argmax(self.predict_proba(X), axis=1)
