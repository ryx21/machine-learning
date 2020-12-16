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
* `gini_impurity`

#### Random Forest Classifier
Builds an ensemble of CART classifiers with random feature selection and random sample selection, currently only using a single CPU process. Optional Parameters (includes also CART parameters):
* `num_trees`
* `max_sample_ratio`