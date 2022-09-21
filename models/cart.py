import numpy as np
from typing import Callable
from models.helpers import label_probabilities, entropy, mean_squared_error, optimal_feature_split, sample_fraction_of_idxs


REGRESSION = 'regression'
CLASSIFICATION = 'classification'

class CART:

    def __init__(self,                 
            max_depth: int=np.infty,
            min_samples_split: int=2,
            max_feature_ratio: float=1.0,
            task: str=CLASSIFICATION,
            custom_loss_function: Callable=None
            ):
        # node depth
        self._depth = 1
        # set up parameters
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._max_feature_ratio = max_feature_ratio
        self._task = task
        self._custom_loss_function = None
        # node information
        self._leaf_value = None
        self._left_child = None
        self._right_child = None
        self._feature_idx = None
        self._feature_threshold = None
        
        if custom_loss_function:
            self._loss_function = custom_loss_function
        elif task == REGRESSION:
            self._loss_function = mean_squared_error
        elif task == CLASSIFICATION:
            self._loss_function = entropy
        else:
            raise AttributeError("Loss function not specified")

    def _split(self, data: np.ndarray, labels: np.array):
        sampled_feature_idxs = sample_fraction_of_idxs(
            data.shape[1], self._max_feature_ratio
        )
        best_loss = np.inf
        for feature_idx in sampled_feature_idxs:
            feature_values = data[:, feature_idx]
            threshold, loss = optimal_feature_split(feature_values, labels, self._loss_function)
            if loss < best_loss:
                best_loss = loss
                self._feature_idx = feature_idx
                self._feature_threshold = threshold

    def _set_leaf_value(self, targets):
        # TODO: ideally this should be tied to the loss function
        if self._task == REGRESSION:
            self._leaf_value = np.mean(targets)
        elif self._task == CLASSIFICATION:
            label_weights = label_probabilities(targets)
            self._leaf_value = np.argmax(label_weights)
        else:
            raise AttributeError("Invalid classifier type")

    def _create_child(self):
        child_node = CART(
            self._max_depth,
            self._min_samples_split,
            self._max_feature_ratio,
            self._task,
            self._custom_loss_function
            )
        child_node._depth = self._depth + 1
        return child_node
        
    def _stop_condition(self, data: np.ndarray, targets: np.array) -> bool:
        # check if fewer than required number of samples per leaf
        cond1 = len(data) < self._min_samples_split
        # check if leaf is pure
        cond2 = np.min(targets) == np.max(targets)
        # check if node depth is at max depth
        cond3 = self._depth == self._max_depth
        conditions = [cond1, cond2, cond3]
        return any(conditions)

    def fit(self, data, targets):

        if self._stop_condition(data, targets):
            self._set_leaf_value(targets)
            return
        else:
            self._split(data, targets)
            mask = data[:, self._feature_idx] <= self._feature_threshold

        if min(mask) == max(mask):
            self._set_leaf_value(targets)
            return
        else:
            # create and fit left child node
            self._left_child = self._create_child()
            self._left_child.fit(data[mask, :], targets[mask])
            # create and fit right child node
            self._right_child = self._create_child()
            self._right_child.fit(data[~mask, :], targets[~mask])

    def predict_sample(self, x):
        # recurse down tree until leaf node is hit
        if self._leaf_value is not None:
            return self._leaf_value
        elif x[self._feature_idx] <= self._feature_threshold:
            return self._left_child.predict_sample(x)
        else:
            return self._right_child.predict_sample(x)
            
    def predict(self, data):
        predictions = []
        for x in data:
            predictions.append(self.predict_sample(x))
        return np.array(predictions)
