from sklearn import datasets


def get_iris_demo_splits():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    n_samples = len(X)

    split = int(0.7 * n_samples)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test