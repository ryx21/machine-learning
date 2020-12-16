from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from models.random_forest import RandomForestClassifier

if __name__ == "__main__":

    np.random.seed(0)

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    n_samples = len(X)
    n_classes = 3

    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    RF_model = RandomForestClassifier(n_classes=n_classes, min_samples_leaf=10, max_depth=5, num_trees=200, max_sample_ratio=0.8)
    RF_model.fit(X_train, y_train)

    y_train_predict = RF_model.predict(X_train)
    y_test_predict = RF_model.predict(X_test)

    print('--- Random Forest Classifier ---')
    print("Train Accuracy:", accuracy_score(y_train, y_train_predict))
    print("Test Accuracy:", accuracy_score(y_test, y_test_predict))