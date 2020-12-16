from numpy.testing import assert_array_equal
from models.helpers import *


def test_get_label_probabilities():
    data_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    result = get_label_probabilities(data_labels, n_classes=4)
    assert_array_equal(result, [3/8, 3/8, 2/8, 0])


def test_gini_impurity():
    data_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    result = gini_impurity(data_labels, n_classes=4)
    assert result == 0.65625


def _test_entropy_impurity():
    # TODO: write test
    pass


def _test_misclassification_impurity():
    # TODO: write test
    pass


def test_get_split_impurity():
    left_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    right_labels = np.array([1, 3, 3, 3])
    result = get_split_impurity(left_labels, right_labels, n_classes=4, impurity_function=gini_impurity)
    assert result == 0.5625


def test_find_best_split_for_feature():
    feature_values = np.array([0, 1, 4, 6, 8, 8, 10])
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    best_threshold, best_impurity = find_best_split_for_feature(
        feature_values, labels, n_classes=2, impurity_function=gini_impurity)
    assert best_threshold == 4
    assert best_impurity == 0.0


