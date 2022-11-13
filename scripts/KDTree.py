from typing import List

import numpy as np


class KDTree:
    axis_num = None

    root = None

    hash_map = dict()

    def build(self, X: np.array, leaf_size: int, depth=0):
        n = depth % self.axis_num
        sorted_X = sorted(X, key=lambda point: point[n])

        len_X = len(X)

        if len_X <= leaf_size * 2 + 1:
            return {
                'point': sorted_X,
                'list': True,
                'left': None,
                'right': None,
                'axis': n
            }

        return {
            'point': sorted_X[len_X // 2],
            'list': False,
            'left': self.build(sorted_X[:len_X // 2], depth + 1),
            'right': self.build(sorted_X[len_X // 2 + 1:], depth + 1),
            'axis': n
        }

    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.axis_num = len(X[0])

        self.hash_map = dict()

        for i, val in enumerate(X):
            self.hash_map[str(val)] = i

        self.root = self.build(X, leaf_size)

    def get_distance(self, point1, point2):
        return np.sum((point1 - point2) ** 2)

    def get_max_distance(self, point1, point_arr):
        maxi = -1
        for point in point_arr:
            maxi = max(maxi, self.get_distance(point1, point))
        return maxi

    def relax(self, pivot, point1, point2, k):
        return sorted(point1 + point2, key=lambda point: self.get_distance(pivot, point))[:k]

    def get_points(self, node, point, k):
        nearest = None
        not_nearest = None

        n = node['axis']

        if not node['list']:
            if point[n] < node['point'][n]:
                nearest = node['left']
                not_nearest = node['right']
            else:
                nearest = node['right']
                not_nearest = node['left']

            tmp_points = self.get_points(nearest, point, k)

            if self.get_max_distance(point, tmp_points) > \
                    (point[n] - node['point'][n]) ** 2:
                tmp_points = self.relax(point, self.get_points(not_nearest, point, k) + [node['point']], tmp_points, k)
            elif len(tmp_points) < k:
                tmp_points = self.relax(point, self.get_points(not_nearest, point, k) + [node['point']], tmp_points, k)
            return tmp_points
        else:
            return self.relax(point, [], node['point'], k)

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        ans = []
        for point in X:
            tmp_arr = []

            for j in self.get_points(self.root, point, k):
                tmp_arr.append(self.hash_map[str(j)])
            ans.append(tmp_arr)
        return ans
