import numpy as np
import math
import logging
import time
from pathos import multiprocessing
from models.cart import CART, CLASSIFICATION

logger = logging.getLogger(__name__)

class RandomForestClassifier:

    def __init__(self,
                 max_depth=np.infty,
                 min_samples_split=2,
                 max_feature_ratio=1,
                 num_trees=100,
                 max_sample_ratio=1,
                 ):
        # RandomForest info
        self._n_classes = None
        self._num_trees = num_trees
        self._base_learner_ensemble = []
        self._max_sample_ratio = max_sample_ratio
        # store base-learner parameters in dictionary
        self._base_learner_args = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_feature_ratio": max_feature_ratio,
            "task": CLASSIFICATION
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
        base_learner = CART(**self._base_learner_args)
        base_learner.fit(data, labels)
        return base_learner

    def _build_ensemble_multi_process(self, data, labels, n_jobs):
        jobs_launched = 0
        def single_job(i):
            nonlocal jobs_launched
            jobs_launched += 1
            logger.info(f"Fitting base learning: {jobs_launched}")
            return self._build_base_learner(data, labels)
        # use maximum number of CPU cores
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        with multiprocessing.Pool(n_jobs) as p:
            self._base_learner_ensemble = p.map(single_job, range(self._num_trees))

    def fit(self, data, labels, n_jobs=1):

        self._n_classes = max(labels) + 1
        start_time = time.time()
        jobs_launched = 0

        def single_job(i):
            nonlocal jobs_launched
            jobs_launched += 1
            if jobs_launched % 10 == 0:
                logger.info(f"Fitting base learner: {jobs_launched}")
            return self._build_base_learner(data, labels)
        
        # use maximum number of CPU cores
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        with multiprocessing.Pool(n_jobs) as p:
            self._base_learner_ensemble = p.map(single_job, range(self._num_trees))
        logger.info(f'Fitted {self._num_trees} base learners in: {round(time.time() - start_time, 3)} seconds')

    def predict_sample_probability(self, x):
        # predict probability of each class for a single sample
        ensemble_votes = np.zeros(self._n_classes)
        for base in self._base_learner_ensemble:
            prediction = base.predict_sample(x)
            ensemble_votes[prediction] += 1
        ensemble_votes = np.array(ensemble_votes) / self._num_trees
        return ensemble_votes

    def predict_probability(self, data):
        assert len(self._base_learner_ensemble) > 0, "Must call fit() first"
        result = np.array([self.predict_sample_probability(x) for x in data])
        return result

    def predict(self, data):
        result = self.predict_probability(data)
        result = np.argmax(result, axis=1)
        return result
