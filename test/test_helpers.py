import pytest
from numpy.testing import assert_array_equal
from models.helpers import *


EPS = 1e-6

@pytest.mark.parametrize('labels, probabilities',
    [
        ([0, 0, 0], [1]),
        ([0, 1, 1, 2, 2, 2], [1/6, 2/6, 3/6]),
        ([1, 1, 4, 4, 4, 6, 6, 6, 6], [0, 2/9, 0, 0, 3/9, 0, 4/9])
    ]
)
def test_label_probabilities(labels, probabilities):
    result = label_probabilities(labels)
    assert_array_equal(result, probabilities)


@pytest.mark.parametrize('targets, expected',
    [
        ([0, 0, 0], 0),
        ([0, 1, 1, 2, 2, 2], 1.4591479170272446),
        ([1, 1, 4, 4, 4, 6, 6, 6, 6], 1.5304930700451949)
    ]
)
def test_entropy(targets, expected):
    assert entropy(targets) == pytest.approx(expected, EPS)


@pytest.mark.parametrize('targets, expected',
    [
        ([0, 0, 0], 0),
        ([2, 3, 4, 5, 6], 51.2)
    ]
)
def test_mean_square_error(targets, expected):
    assert mean_squared_error(targets) == pytest.approx(expected, EPS)


def test_optimal_feature_split_classification_case():
    features = np.array([0, 1, 4, 6, 8, 8, 10])
    targets = np.array([0, 0, 0, 1, 1, 1, 1])
    best_threshold, best_impurity = optimal_feature_split(features, targets, loss_function=lambda x: len(np.unique(x)))
    assert best_threshold == 4
    assert best_impurity == 1.0


def test_optimal_feature_split_regression_case():
    features = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    targets = np.array([-11, -10, -9, 9, 10, 11, 12, 13])
    best_threshold, best_impurity = optimal_feature_split(features, targets, loss_function=lambda x: np.std(x))
    assert best_threshold == 3
    assert best_impurity == pytest.approx(1.1900696943310818, EPS)


@pytest.mark.parametrize('left, right, loss_func,expected',
    [
        (np.array([0, 0, 0, 1, 1, 1, 2, 2]), np.array([1, 3, 3, 3]), lambda x: np.sum(x), 8),
        (np.array([0, 0, 1]), np.array([]), lambda x: len(x), 3)
    ]
)
def test_weighted_split_loss(left, right, loss_func, expected):
    result = weighted_split_loss(left, right, loss_func)
    assert result == pytest.approx(expected, 0.0001)
