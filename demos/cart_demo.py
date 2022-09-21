from sklearn.metrics import accuracy_score
from models.cart import CART
from demo_utils import get_iris_demo_splits


X_train, y_train, X_test, y_test = get_iris_demo_splits()

CART_model = CART(max_depth=3)
CART_model.fit(X_train, y_train)

y_train_predict = CART_model.predict(X_train)
y_test_predict = CART_model.predict(X_test)

print('--- CART Classifier ---')
print("Train Accuracy:", accuracy_score(y_train, y_train_predict))
print("Test Accuracy:", accuracy_score(y_test, y_test_predict))
