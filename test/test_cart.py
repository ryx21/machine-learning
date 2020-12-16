from models.cart import CART
import numpy as np
import pytest


test_check_arguments_args = [
    {"n_classes": 1},
    {"n_classes": 2, "max_depth": 0},
    {"n_classes": 2, "min_samples_leaf": 0},
    {"n_classes": 2, "min_samples_split": 1},
    {"n_classes": 2, "max_feature_ratio": 1.1},
    {"n_classes": 2, "max_feature_ratio": -0.1},
    {"n_classes": 2, "split_criterion": "not_gini"}
]

test_stop_condition_args = [
    (np.array([[1, 1], [1, 1]]), np.array([1, 1])),  # np.min(labels) == np.max(labels)
    (np.array([[0, 0]]), np.array([0]))              # len(data) < self._min_samples_split
]


@pytest.mark.parametrize('cart_kwargs', test_check_arguments_args)
def test_check_arguments(cart_kwargs):
    # test check_arguments raises catches flags illegal arguments to CART
    with pytest.raises(AssertionError):
        CART(**cart_kwargs)


def test_sample_features():
    cart_test = CART(n_classes=2, max_feature_ratio=0.5)
    assert len(np.unique(cart_test._sample_features(10))) == 5


def test_sample_features_close_to_zero_case():
    cart_test = CART(n_classes=2, max_feature_ratio=0.0001)
    assert len(np.unique(cart_test._sample_features(10))) == 1


def test_sample_features_close_to_one_case():
    cart_test = CART(n_classes=2, max_feature_ratio=0.9999)
    assert len(np.unique(cart_test._sample_features(10))) == 10


def test_create_child():
    parent_node = CART(n_classes=2)
    child_node = parent_node._create_child()
    assert parent_node._n_classes == child_node._n_classes
    assert parent_node._depth + 1 == child_node._depth
    assert parent_node._max_depth == child_node._max_depth
    assert parent_node._min_samples_leaf == child_node._min_samples_leaf
    assert parent_node._min_samples_split == child_node._min_samples_split
    assert parent_node._split_criterion == child_node._split_criterion
    assert parent_node._max_feature_ratio == child_node._max_feature_ratio


@pytest.mark.parametrize("data, labels", test_stop_condition_args)
def test_stop_condition(data, labels):
    cart_test = CART(n_classes=2)
    assert cart_test._stop_condition(data, labels)


def test_stop_condition_max_depth_case():
    parent_node = CART(n_classes=2, max_depth=2)
    child_node = parent_node._create_child()
    data = np.array([[1, 1], [1, 1]])
    labels = np.array([1, 0])
    # check parent (depth=1) doesn't stop but child (depth=2) does stop
    assert not parent_node._stop_condition(data, labels)
    assert child_node._stop_condition(data, labels)
