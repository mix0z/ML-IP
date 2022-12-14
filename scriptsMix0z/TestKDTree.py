"""Tests for `kdtree` package."""
import numpy as np
from KDTree import KDTree


def true_closest(X_train, X_test, k):
    """Find the true closest points in X_train to each point in X_test."""
    result = []
    for x0 in X_test:
        bests = list(
            sorted(
                [(i, np.linalg.norm(x - x0)) for i, x in enumerate(X_train)],
                key=lambda x: x[1],
            )
        )
        bests = [i for i, d in bests]
        result.append(bests[: min(k, len(bests))])
    return result


X_train = np.random.randn(100, 15)
X_test = np.random.randn(10, 15)
tree = KDTree(X_train, leaf_size=10)
predicted = tree.query(X_test, k=5)
true = true_closest(X_train, X_test, k=5)

if np.sum(np.abs(np.array(np.array(predicted).shape) - np.array(np.array(true).shape))) != 0:
    print("Wrong shape")
else:
    errors = sum([1 for row1, row2 in zip(predicted, true) for i1, i2 in zip(row1, row2) if i1 != i2])
    if errors > 0:
        print("Encounted", errors, "errors")
