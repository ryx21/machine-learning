from demo_utils import get_iris_demo_splits
import numpy as np
from sklearn.metrics import accuracy_score
from models.random_forest import RandomForestClassifier


X_train, y_train, X_test, y_test = get_iris_demo_splits()

RF_model = RandomForestClassifier(
    min_samples_split=3,
    max_depth=4,
    num_trees=100,
    max_sample_ratio=0.5,
)
RF_model.fit(X_train, y_train, n_jobs=-1)

y_train_predict = RF_model.predict(X_train)
y_test_predict = RF_model.predict(X_test)

print('--- Random Forest Classifier ---')
print("Train Accuracy:", accuracy_score(y_train, y_train_predict))
print("Test Accuracy:", accuracy_score(y_test, y_test_predict))