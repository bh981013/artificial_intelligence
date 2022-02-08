import matplotlib.pyplot as plt
import time
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False
)
f = open("output.txt", "w")
for a in range(1, 10):
    knn1 = KNeighborsClassifier(n_neighbors=a)
    knn1.fit(X_train, y_train)
    predicted_knn = knn1.predict(X_test)
    d = metrics.classification_report(y_test, predicted_knn, output_dict = True)
    f.write(str(d["accuracy"]) + "\n")
