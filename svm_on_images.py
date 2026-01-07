import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

digits = load_digits()
X = digits.images
y = digits.target


X = X.reshape((X.shape[0], -1))


X = X / X.max()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


classifier = svm.SVC(kernel="rbf", gamma=0.01)

classifier.fit(X_train, y_train)


predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("SVM on raw pixels Accuracy:", accuracy)
