from sklearn import datasets
from sklearn.metrics import accuracy_score
from models.cart import CART

if __name__ == "__main__":

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    n_samples = len(X)
    n_classes = 3

    split = int(0.7 * n_samples)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    CART_model = CART(n_classes=n_classes, max_depth=3)
    CART_model.fit(X_train, y_train)

    y_train_predict = CART_model.predict(X_train)
    y_test_predict = CART_model.predict(X_test)

    print('--- CART Classifier ---')
    print("Train Accuracy:", accuracy_score(y_train, y_train_predict))
    print("Test Accuracy:", accuracy_score(y_test, y_test_predict))
