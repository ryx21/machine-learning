import numpy as np
import math
from typing import Callable


def label_probabilities(data_labels: np.array):
    label_counts = np.zeros(max(data_labels)+1)
    for label in data_labels:
        label_counts[label] += 1
    label_probabilities = label_counts / len(data_labels)
    return label_probabilities


def sample_fraction_of_idxs(n_max: int, fraction: int):
    num_samples = math.ceil(n_max * fraction)
    sampled_features = np.random.choice(
        list(range(n_max)),
        size=num_samples,
        replace=False)
    return np.sort(sampled_features)


def entropy(targets: np.array):
    if not len(targets):
        return 0
    eps = 1e-10
    probabilities = label_probabilities(targets)
    # ensure we don't take logs of 0s
    probabilities[probabilities < eps] = eps
    result = -np.dot(probabilities, np.log2(probabilities))
    return result


def mean_squared_error(targets: np.array):
    if not len(targets):
        return 0
    return  (np.sum(targets) - np.mean(targets))**2 / len(targets)


def weighted_split_loss(left_targets: np.array, right_targets: np.array, loss_function: Callable) -> float:
    n = len(left_targets) + len(right_targets)
    weighted_left_split_loss = len(left_targets) / n * loss_function(left_targets)
    weighted_right_split_loss = len(right_targets) / n * loss_function(right_targets)
    return weighted_left_split_loss + weighted_right_split_loss


def optimal_feature_split(feature_values: np.array, targets: np.array, loss_function: Callable):
    # sort labels by feature_values
    sort_indices = np.argsort(feature_values)
    sorted_feature_values = feature_values[sort_indices]
    sorted_targets = targets[sort_indices]
    best_loss = np.inf
    # find optimum threshold that minimises loss function
    for i in range(len(sorted_feature_values) - 1):
        loss = weighted_split_loss(sorted_targets[:i+1], sorted_targets[i+1:], loss_function)
        if loss < best_loss:
            best_loss = loss
            best_threshold = sorted_feature_values[i]
    return best_threshold, best_loss