import numpy as np


def get_label_probabilities(data_labels, n_classes):
    label_counts = np.zeros(n_classes)
    for label in data_labels:
        label_counts[label] += 1
    label_probabilities = label_counts / len(data_labels)
    return label_probabilities


def gini_impurity(data_labels, n_classes):
    label_probabilities = get_label_probabilities(data_labels, n_classes)
    result = np.dot(label_probabilities, 1-label_probabilities)
    return result


def get_split_impurity(left_labels, right_labels, n_classes, impurity_function):
    sample_size = len(left_labels) + len(right_labels)
    left_split_impurity = impurity_function(left_labels, n_classes)
    right_split_impurity = impurity_function(right_labels, n_classes)
    result = len(left_labels)/sample_size * left_split_impurity + len(right_labels)/sample_size* right_split_impurity
    return result


def find_best_split_for_feature(feature_values, labels, n_classes, impurity_function=gini_impurity):
    # sort labels by feature_values
    sort_indices = np.argsort(feature_values)
    sorted_feature_values = feature_values[sort_indices]
    sorted_labels = labels[sort_indices]
    # find optimum threshold that minimises impurity function
    for i in range(len(sorted_feature_values) - 1):
        split_impurity = get_split_impurity(
                sorted_labels[:i+1],
                sorted_labels[i+1:],
                n_classes,
                impurity_function)
        if i == 0:
            best_impurity = split_impurity
            best_threshold = sorted_feature_values[i]
        elif split_impurity < best_impurity:
            best_impurity = split_impurity
            best_threshold = sorted_feature_values[i]
    return best_threshold, best_impurity
