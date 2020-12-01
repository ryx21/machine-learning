from models.helpers import *

# TODO: implement min_samples_leaf
# TODO: implement alternative splitting functions
# TODO: implement regression tree


class CART:

    def __init__(self, n_classes, depth=0, **kwargs):
        self._n_classes = n_classes
        self._depth = depth
        # set up kwargs
        self._config = kwargs
        self._max_depth = kwargs.get('max_depth', np.infty)
        self._min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self._min_samples_split = kwargs.get('min_samples_split', 2)
        self._impurity_function = kwargs.get('impurity_function', gini_impurity)
        # node information
        self._leaf_class = None
        self._is_leaf_node = False
        self._left_child = None
        self._right_child = None
        self._feature_index = None
        self._feature_threshold = None

    def _find_optimum_split(self, data, labels):
        # data (#samples, #features), labels (#samples)
        # find best feature to split on by impurity
        for i in range(data.shape[1]):
            feature_values = data[:, i]
            feature_threshold, feature_impurity = find_best_split_for_feature(
                feature_values, labels, self._n_classes, self._impurity_function)
            if i == 0:
                best_feature_threshold = feature_threshold
                best_feature_impurity = feature_impurity
                best_feature_index = i
            elif feature_impurity < best_feature_impurity:
                best_feature_threshold = feature_threshold
                best_feature_impurity = feature_impurity
                best_feature_index = i
        # update node with feature threshold and feature index
        self._feature_threshold = best_feature_threshold
        self._feature_index = best_feature_index

    def _create_leaf(self, labels):
        self._is_leaf_node = True
        self._leaf_class = np.argmax(get_label_probabilities(labels, self._n_classes))

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
            # create and fit left child node
            self._left_child = CART(self._n_classes, self._depth + 1, **self._config)
            self._left_child.fit(data[mask, :], labels[mask])
            # create and fit right child node
            self._right_child = CART(self._n_classes, self._depth + 1, **self._config)
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

    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            prediction = self.predict_sample(x[i])
            predictions.append(prediction)
        result = np.array(predictions)
        return result
