from models.cart import CART, CLASSIFICATION, REGRESSION
import pytest
import numpy as np

np.random.seed(2)

@pytest.mark.parametrize("data, targets, expected",
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), np.array([1, 1, 1]), True),
        (np.array([[0, 0], [0, 0]]), np.array([0, 1]), True),
        (np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), np.array([0, 0, 1, 1]), False)
    ]
)
def test_stop_condition(data, targets, expected):
    cart_test = CART(min_samples_split=3, max_depth=3)
    assert cart_test._stop_condition(data, targets) == expected


@pytest.mark.parametrize("max_depth, depth, expected",
    [
        (2, 2, True),
        (10, 8, False)
    ]
)
def test_stop_condition_max_depth_case(max_depth, depth, expected):
    cart_test = CART(max_depth=max_depth, min_samples_split=2)
    cart_test._depth = depth
    data = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    labels = np.array([0, 0, 1, 1])
    assert cart_test._stop_condition(data, labels) == expected


@pytest.mark.parametrize("data, targets, cart",
    [
        (np.random.normal(size=(100, 10)), np.random.normal(size=(100)), CART(max_depth=5, task=REGRESSION)),
        (np.random.normal(size=(100, 10)), np.random.choice([0, 1, 2, 3], size=(100)), CART(max_depth=5, task=CLASSIFICATION)),
    ]
)
def test_fuzzy_fit(data, targets, cart):
    cart.fit(data, targets)
    assert True