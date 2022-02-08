import matplotlib.pyplot as plt
import time
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

digits = datasets.load_digits()

clf = [svm.SVC(gamma=0.001, C = 10) for _ in range(3)]
knn = [KNeighborsClassifier(n_neighbors=3) for _ in range(3)]
clf_linear = [linear_model.LogisticRegression(max_iter=10000) for _ in range(3)]

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    digits.data, digits.target, train_size=300, shuffle=False
)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    digits.data, digits.target, train_size=600, shuffle=False
)

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    digits.data, digits.target, train_size=1200, shuffle=False
)

predicted = []

clf[0].fit(X_train1, y_train1)
predicted.append(clf[0].predict(X_test3))

clf[1].fit(X_train2, y_train2)
predicted.append(clf[1].predict(X_test3))

clf[2].fit(X_train2, y_train2)
predicted.append(clf[2].predict(X_test3))


predicted_linear = []

clf_linear[0].fit(X_train1, y_train1)
predicted_linear.append(clf_linear[0].predict(X_test3))

clf_linear[1].fit(X_train2, y_train2)
predicted_linear.append(clf_linear[1].predict(X_test3))

clf_linear[2].fit(X_train3, y_train3)
predicted_linear.append(clf_linear[2].predict(X_test3))


predicted_knn = []

knn[0].fit(X_train1, y_train1)
predicted_knn.append(knn[0].predict(X_test3))

knn[1].fit(X_train2, y_train2)
predicted_knn.append(knn[1].predict(X_test3))

knn[2].fit(X_train3, y_train3)
predicted_knn.append(knn[2].predict(X_test3))

ytest = [y_test3, y_test3, y_test3]

f = open("accuracy_output4.txt", "w")

arr =[100, 600, 1200]
name  = ["svm", "logistic", "KNN"]
algorithm = [predicted, predicted_linear, predicted_knn]
for j in range(3):
    f.write(name[j] + "\n")
    for i in range(3):
        disp_knn= metrics.ConfusionMatrixDisplay.from_predictions(ytest[0], algorithm[j][i])
        disp_knn.figure_.suptitle("knn Confusion Matrix")
        # f.write(f"Confusion matrix:\n{disp_knn.confusion_matrix}\n")

        d = metrics.classification_report(ytest[i], algorithm[j][i], output_dict = True)
        # f.write(metrics.classification_report(ytest[i], algorithm[j][i]))
        f.write(str(arr[i]) + " " + str(d["accuracy"]) + "\n")