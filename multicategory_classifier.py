from logistic_classifier import logistic_classification, predict, add_bias
import matplotlib.pyplot as plt
from typing import List

import numpy as np

test_features = np.array([
    [0.5, 1.0], [1.0, 0.5], [1.5, 0.0], [2.0, -0.5], [2.5, -1.0],
    [-0.5, 1.5], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.5, -0.5],

    [1.0, 3.0], [1.5, 2.5], [2.0, 2.0], [2.5, 1.5], [3.0, 1.0],
    [1.5, 3.5], [2.0, 3.0], [2.5, 2.5], [3.0, 2.0], [3.5, 1.5],

    [-1.0, 3.0], [-0.5, 3.5], [0.0, 3.0], [0.5, 3.5], [1.0, 4.0],
    [-1.5, 2.5], [-1.0, 2.0], [-0.5, 2.5], [0.0, 2.0], [0.5, 2.5]
])
noise = np.random.uniform(-1, 1, test_features.shape)
test_features += noise

test_categories = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

def one_hot_encode(categories: np.ndarray) -> np.ndarray:
    categories = np.eye(np.max(categories)+1)[categories]
    return categories

def classify_categories(classifiers: List[np.ndarray],
                        features: np.ndarray,) -> np.ndarray:
    predictions = np.zeros(shape=(features.shape[0], len(classifiers)))
    for i, weights in enumerate(classifiers):
        predictions[:, i] = predict(weights, add_bias(features))

    def de_encode(row):
        row = np.argmax(row)
        return row
    predictions = np.apply_along_axis(de_encode, 1, predictions)
    return predictions



def multicategory_classification(
        features: np.ndarray,
        categories: np.ndarray,
        step: float, steps: int) -> List[np.ndarray]:
    categories = one_hot_encode(categories)
    classifiers = []
    for category in categories.T:
        classifiers.append(logistic_classification(features, category, step, steps))
    return classifiers


def plot_multicategory_classification(classifiers: List[np.ndarray],
                        features: np.ndarray,
                        categories: np.ndarray):
    plt.figure(figsize=(10, 6))
    x = features[:, 0]
    y = features[:, 1]
    plt.scatter(x[categories == 0], y[categories == 0], color='blue')
    plt.scatter(x[categories == 1], y[categories == 1], color='red')
    plt.scatter(x[categories == 2], y[categories == 2], color='green')
    for weights in classifiers:
        line = [-weights[0]/weights[1], -weights[2]/weights[1]]
        if line is not None:
            x_min, x_max = min(x), max(x)
            y_min = line[0] * x_min + line[1]
            y_max = line[0] * x_max + line[1]
            plt.plot([x_min, x_max], [y_min, y_max], color='black', linewidth=2)
    plt.title('Point color classification', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    classifiers = multicategory_classification(test_features, test_categories, 0.01, 10000)
    plot_multicategory_classification(classifiers, test_features, test_categories)
    print(classify_categories(classifiers, test_features))

