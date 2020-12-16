import numpy as np
import math
from models.helpers import find_best_split_for_feature, get_label_probabilities, gini_impurity

# TODO: implement min_samples_leaf
# TODO: implement alternative splitting functions
# TODO: implement regression tree


class CART:

    def __init__(self,
                 n_classes,
                 depth=1,
                 max_depth=np.infty,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 split_criterion="gini",
                 max_feature_ratio=1,
                 ):
        self._n_classes = n_classes
        self._depth = depth
        # set up parameters
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_split = min_samples_split
        self._split_criterion = split_criterion
        self._max_feature_ratio = max_feature_ratio
        # node information
        self._leaf_class = None
        self._is_leaf_node = False
        self._left_child = None
        self._right_child = None
        self._feature_index = None
        self._feature_threshold = None
        # assert all arguments are allowable
        self._check_arguments()
        # setup impurity function
        self._impurity_function = self._get_impurity_function(split_criterion)

    def _check_arguments(self):
        assert self._n_classes > 1, "'n_classes' must be greater than 1"
        assert self._max_depth > 0, "'max_depth' must be greater than 0"
        assert self._min_samples_leaf > 0, "'min_samples_leaf' must be greater than 0"
        assert self._min_samples_split > 1, "'min_samples_split' must be greater than 1"
        assert (1 >= self._max_feature_ratio > 0), "'max_feature_ratio' must be within [0, 1)"
        assert self._split_criterion in ['gini', 'entropy', 'misclassification'],\
            "'split_criterion' must be one of: gini, entropy, misclassification'"

    @staticmethod
    def _get_impurity_function(impurity_function):
        if impurity_function == "gini":
            return gini_impurity
        elif impurity_function == "entropy":
            raise NotImplementedError
        elif impurity_function == "misclassification":
            raise NotImplementedError

    def _sample_features(self, num_features):
        # get number of sampled features
        num_sampled_features = math.ceil(num_features * self._max_feature_ratio)
        sampled_features = np.random.choice(list(range(num_features)), size=num_sampled_features, replace=False)
        return np.sort(sampled_features)

    def _find_optimum_split(self, data, labels):
        # data (#samples, #features), labels (#samples)
        # randomly sample features
        sampled_features = self._sample_features(num_features=data.shape[1])
        # find best feature to split on by impurity
        for i, feature in enumerate(sampled_features):
            feature_values = data[:, feature]
            feature_threshold, feature_impurity = find_best_split_for_feature(
                feature_values, labels, self._n_classes, self._impurity_function)
            if i == 0:
                best_feature_threshold = feature_threshold
                best_feature_impurity = feature_impurity
                best_feature_index = feature
            elif feature_impurity < best_feature_impurity:
                best_feature_threshold = feature_threshold
                best_feature_impurity = feature_impurity
                best_feature_index = feature
        # update node with feature threshold and feature index
        self._feature_threshold = best_feature_threshold
        self._feature_index = best_feature_index

    def _create_leaf(self, labels):
        self._is_leaf_node = True
        self._leaf_class = np.argmax(get_label_probabilities(labels, self._n_classes))

    def _create_child(self):
        child_node = CART(
            self._n_classes,
            self._depth + 1,
            self._max_depth,
            self._min_samples_leaf,
            self._min_samples_split,
            self._split_criterion,
            self._max_feature_ratio)
        return child_node

    def _stop_condition(self, data, labels):
        # check if fewer than required number of samples per leaf
        cond1 = len(data) < self._min_samples_split
        # check if leaf is pure
        cond2 = np.min(labels) == np.max(labels)
        # check if node depth is at max depth
        cond3 = self._depth == self._max_depth
        conditions = [cond1, cond2, cond3]
        return any(conditions)

    def fit(self, data, labels):
        if not self._stop_condition(data, labels):
            self._find_optimum_split(data, labels)
            # split data and creates left and right child nodes
            mask = data[:, self._feature_index] <= self._feature_threshold
            # checks the split actually partitions data, otherwise make leaf
            if np.min(mask) == np.max(mask):
                self._create_leaf(labels)
            else:
                # create and fit left child node
                self._left_child = self._create_child()
                self._left_child.fit(data[mask, :], labels[mask])
                # create and fit right child node
                self._right_child = self._create_child()
                self._right_child.fit(data[~mask, :], labels[~mask])
        else:
            self._create_leaf(labels)

    def predict_sample(self, x):
        # recurse down tree until leaf node is hit
        if self._is_leaf_node is False:
            if x[self._feature_index] <= self._feature_threshold:
                return self._left_child.predict_sample(x)
            else:
                return self._right_child.predict_sample(x)
        # return leaf node prediction
        else:
            return self._leaf_class

    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            prediction = self.predict_sample(data[i])
            predictions.append(prediction)
        result = np.array(predictions)
        return result
