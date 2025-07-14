from logistic_classifier import logistic_classification, predict, add_bias

import numpy as np

test_features = np.array([
    [0.5, 1.0], [1.0, 0.5], [1.5, 0.0], [2.0, -0.5], [2.5, -1.0],
    [-0.5, 1.5], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.5, -0.5],

    [1.0, 3.0], [1.5, 2.5], [2.0, 2.0], [2.5, 1.5], [3.0, 1.0],
    [1.5, 3.5], [2.0, 3.0], [2.5, 2.5], [3.0, 2.0], [3.5, 1.5],

    [-1.0, 3.0], [-0.5, 3.5], [0.0, 3.0], [0.5, 3.5], [1.0, 4.0],
    [-1.5, 2.5], [-1.0, 2.0], [-0.5, 2.5], [0.0, 2.0], [0.5, 2.5]
])

test_categories = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

def one_hot_encode(categories: np.ndarray):
    categories = np.eye(np.max(categories)+1)[categories]


def multicategorical_classification