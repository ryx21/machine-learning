# Algorithms

My implementations of machine learning algorithms.
* CART (Classification and Regression Tree) Algorithm
* Random Forest classifier

### Project Setup
1. Setup python path
```
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```
2. Install dependencies
```
pip install -r requirements.txt
```
`models/` contains the algorithm implementations, `tests/` contain unit tests, and `demos` contain demo scripts for each algorithm.
### Models
#### CART Algorithm
The CART implementation currently supports only categorical target variables and numerical features. 
Optional parameters:
* `max_depth`
* `min_samples_split`
* `max_feature_ratio`

Supported splitting functions:
* `gini`
* `entropy`
* `misclassification`

#### Random Forest Classifier
Builds an ensemble of CART classifiers with random feature selection and random sample selection. Optional Parameters (includes also CART parameters):
* `num_trees`
* `max_sample_ratio`

Call `fit(data, labels, n_jobs)` function to fit model to train model. `n_jobs` can be used to set number of multiprocessing jobs, (`n_jobs = -1` uses all available cores). By default `n_jobs=1` which uses a single process.