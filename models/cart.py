import numpy as np
from models.helpers import *

# TODO: implement inference
# TODO: implement stop conditions properly ZZZ


class CARTNode:

    def __init__(self, data, labels, n_classes, depth=0, **kwargs):
        self.n_classes = n_classes
        # set up kwargs
        self.config = kwargs
        self.max_depth = kwargs.get('max_depth', np.infty)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.impurity_function = kwargs.get('impurity_function', gini_impurity)
        # node information
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.feature_index = None
        self.feature_threshold = None
        # grow tree
        self._grow_tree(data, labels, self.config)

    def _find_optimum_split(self, data, labels):
        # data (#samples, #features), labels (#samples)
        # find best feature to split on by impurity
        for i in range(data.shape[1]):
            feature_values = data[:, i]
            feature_threshold, feature_impurity = find_best_split_for_feature(
                feature_values, labels, self.n_classes, self.impurity_function)
            if i == 0:
                best_feature_threshold = feature_threshold
                best_feature_impurity = feature_impurity
                best_feature_index = i
            elif feature_impurity < best_feature_impurity:
                best_feature_threshold = feature_threshold
                best_feature_impurity = feature_impurity
                best_feature_index = i
        # update node with feature threshold and feature index
        self.feature_threshold = best_feature_threshold
        self.feature_index = best_feature_index

    def _stop_condition(self, labels, data):
        # check if fewer than required number of samples per leaf # TODO: this should be done before split...?
        cond1 = len(data) < self.min_samples_leaf
        # check if leaf is pure
        cond2 = np.min(labels) == np.max(labels)
        # check if node depth is at max depth
        cond3 = self.depth = self.max_depth
        return any(cond1, cond2, cond3)

    def _grow_tree(self, data, labels):
        self._find_optimum_split()
        if self._stop_condition():
            pass
        else:
            # split data and creates left and right child nodes
            mask = data[:, self.feature_index] <= self.feature_threshold
            self.left_child = CARTNode(data[mask, :], labels[mask], self.n_classes, self.depth+1, **self.config)
            self.right_child = CARTNode(data[mask, :], labels[mask], self.n_classes, self.depth+1, **self.config)






