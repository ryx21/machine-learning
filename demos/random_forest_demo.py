from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from models.random_forest import RandomForestClassifier

if __name__ == "__main__":

    n_samples = 500
    n_classes = 5
    n_features = 5
    split = int(0.7 * n_samples)

    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=0)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    RF_model = RandomForestClassifier(n_classes=n_classes, max_depth=3, num_trees=10, max_sample_ratio=0.5)
    RF_model.fit(X_train, y_train)

    y_train_predict = RF_model.predict(X_train)
    y_test_predict = RF_model.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_train_predict))
    print("Test Accuracy:", accuracy_score(y_test, y_test_predict))