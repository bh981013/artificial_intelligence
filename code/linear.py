import matplotlib.pyplot as plt
import time
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
digits = datasets.load_digits()

for i in range(5):X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False
)
T = 0
for i in range(5):
    linear = linear_model.LogisticRegression(max_iter=10**i)
    linear_time = time.time()
    linear.fit(X_train, y_train)
    linear_time = time.time() - linear_time
    p = linear.predict(X_test)
    d = metrics.classification_report(y_test, p, output_dict = True)
    print(d["accuracy"])
    print()
