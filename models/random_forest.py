import numpy as np
import math
import time
from pathos import multiprocessing
from models.cart import CART

# TODO: add multiprocessing fit() option
# TODO: write test cases


class RandomForestClassifier:

    def __init__(self,
                 n_classes,
                 max_depth=np.infty,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 split_criterion="gini",
                 max_feature_ratio=1,
                 num_trees=100,
                 max_sample_ratio=1,
                 verbose=True,
                 ):
        # RandomForest info
        self._n_classes = n_classes
        self._num_trees = num_trees
        self._base_learner_ensemble = []
        self._max_sample_ratio = max_sample_ratio
        self._verbose = verbose
        # store base-learner parameters in dictionary
        self._base_learner_args = {
            "n_classes": n_classes,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "split_criterion": split_criterion,
            "max_feature_ratio": max_feature_ratio,
        }

    def _sample_data(self, data, labels):
        num_samples = math.ceil(self._max_sample_ratio * len(data))
        sample_indices = np.random.choice(list(range(len(data))), size=num_samples, replace=False)
        data = data[sample_indices]
        labels = labels[sample_indices]
        return data, labels

    def _build_base_learner(self, data, labels):
        # sample data and labels
        if self._max_sample_ratio < 1:
            data, labels = self._sample_data(data, labels)
        # fit base learner from sampled data
        base_learner = CART(**self._base_learner_args)
        base_learner.fit(data, labels)
        return base_learner

    def _build_ensemble_single_process(self, data, labels):
        for i in range(self._num_trees):
            self._base_learner_ensemble.append(self._build_base_learner(data, labels))
            if self._verbose:
                print(f'--- No. base learners trained: {i + 1} ---')

    def _build_ensemble_multi_process(self, data, labels, n_jobs):
        # build random forest with multiprocessing
        def single_job(i): return self._build_base_learner(data, labels)
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        with multiprocessing.Pool(n_jobs) as p:
            self._base_learner_ensemble = p.map(single_job, range(self._num_trees))

    def fit(self, data, labels, n_jobs=1):
        start_time = time.time()
        if n_jobs != 1:
            self._build_ensemble_multi_process(data, labels, n_jobs)
        else:
            self._build_ensemble_single_process(data, labels)
        if self._verbose:
            print(f'--- Completed training in {round(time.time() - start_time, 3)} seconds ---')

    def predict_sample_probability(self, x):
        ensemble_votes = np.zeros(self._n_classes)
        for base in self._base_learner_ensemble:
            ensemble_votes[base.predict_sample(x)] += 1
        ensemble_votes = np.array(ensemble_votes) / self._num_trees
        return ensemble_votes

    def predict_probability(self, data):
        assert len(self._base_learner_ensemble) > 0, "Must call fit() first"
        result = np.array([self.predict_sample_probability(x) for x in data])
        return result

    def predict(self, x):
        result = self.predict_probability(x)
        result = np.argmax(result, axis=1)
        return result
