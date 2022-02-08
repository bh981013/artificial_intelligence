import matplotlib.pyplot as plt
import time
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
digits = datasets.load_digits()

SVM = svm.SVC(gamma=0.001)
knn = KNeighborsClassifier(n_neighbors=3)

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False
)
T = 0
# for _ in range(100):
#     svm_time = time.time()
#     SVM.fit(X_train, y_train)
#     svm_time =  time.time() - svm_time
#     T += svm_time
# print(T/100)
print("loin: ")
for i in range(5):
    linear = linear_model.LogisticRegression(max_iter=10**i)
    linear_time = time.time()
    linear.fit(X_train, y_train)
    linear_time = time.time() - linear_time
    print(linear_time)
    # T += linear_time
# print(T/100)
# linear_time = time.time()
# linear.fit(X_train, y_train)
# clf_time = time.time() - linear_time

# knn_time  =time.time()
# knn.fit(X_train, y_train)
# knn_time = time.time() - knn_time
