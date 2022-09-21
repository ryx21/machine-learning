import numpy as np
from models.random_forest import RandomForestClassifier


def get_trained_rf(n_samples, n_trees, n_features):
    test_data = np.random.normal(size=(n_samples, n_features))
    test_labels = np.random.binomial(n=1, p=0.1, size=n_samples)
    rf_test = RandomForestClassifier(num_trees=n_trees, max_depth=3)
    rf_test.fit(test_data, test_labels)
    return rf_test


def test_sample_data():
    n_samples = 1000
    sample_ratio = 0.5
    test_data = np.random.normal(size=(n_samples, 4))
    test_labels = np.random.binomial(n=1, p=0.1, size=n_samples)
    rf_test = RandomForestClassifier(max_sample_ratio=sample_ratio)
    sampled_data, sampled_labels = rf_test._sample_data(test_data, test_labels)
    # test number of samples within 10% of expected value
    assert n_samples * 0.5 * 0.9 < len(sampled_data) < n_samples * 0.5 * 1.1
    assert n_samples * 0.5 * 0.9 < len(sampled_labels) < n_samples * 0.5 * 1.1


def test_fit_multi_process():
    n_trees = 42
    test_data = np.random.normal(size=(100, 4))
    test_labels = np.random.binomial(n=1, p=0.1, size=100)
    rf_test = RandomForestClassifier(num_trees=n_trees, max_depth=3)
    rf_test._build_ensemble_multi_process(test_data, test_labels, n_jobs=-1)
    assert len(rf_test._base_learner_ensemble) == n_trees


def test_sample_predict_probability_sum_to_one():
    rf_test = get_trained_rf(n_samples=100, n_trees=42, n_features=4)
    predicted_probabilities = rf_test.predict_sample_probability(np.random.normal(size=4))
    assert np.sum(predicted_probabilities) == 1


def test_predict_probability_shape():
    rf_test = get_trained_rf(n_samples=100, n_trees=42, n_features=4)
    input_shape = (10, 4)
    predicted_probabilities = rf_test.predict_probability(np.random.normal(size=input_shape))
    assert predicted_probabilities.shape == (10, 2)
