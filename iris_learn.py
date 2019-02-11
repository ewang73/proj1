from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import datasets
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


#Read data in
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X, X_dum, Y, Y_dum = train_test_split(X, Y, test_size=0, random_state= 10)
# print(iris.filename)
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)
# print(Y_train)
# print(Y_test)

# DT Find best hyperparameters, plot_learning_curve
t0= time.time()
param_grid = {'max_depth': [i for i in range(2, 20)], 'max_features': [1, 2, 3, 4]}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Decision Tree", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

with open("iris_graphs/irisTree.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)


# NN Find best hyperparameters, plot_learning_curve
t0= time.time()
param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,)], 'learning_rate': ['constant', 'invscaling']}
clf = GridSearchCV(MLPClassifier(solver='sgd', max_iter=50), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Neural Net 50itr", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

t0= time.time()
param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,)], 'learning_rate': ['constant', 'invscaling']}
clf = GridSearchCV(MLPClassifier(solver='sgd', max_iter=200), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Neural Net 200itr", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

t0= time.time()
param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,)], 'learning_rate': ['constant', 'invscaling']}
clf = GridSearchCV(MLPClassifier(solver='sgd', max_iter=10000), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Neural Net Converged", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()



# Boosting Find best hyperparameters, plot_learning_curve
t0= time.time()
param_grid = {'n_estimators': [50, 100, 150, 200, 250, 2000], 'learning_rate': [1., .5, .1]}
clf = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, max_features=3)), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Boosting Decision Trees", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()




# SVM Find best hyperparameters, plot_learning_curve
t0= time.time()
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
degrees = [1,2,3,4,5]
param_grid = [{'kernel': ['linear', 'rbf'], 'C': Cs, 'gamma': gammas},
              {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}]

clf = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = SVC(C=1, gamma=0.001, kernel = 'linear')
clf = clf.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: SVM", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()




# KNN Find best hyperparameters, plot_learning_curve
t0= time.time()
param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7]}
clf = GridSearchCV(KNeighborsClassifier(weights='distance', metric='euclidean'), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
# print(clf.cv_results_)
# print('----------------')
print(clf.best_params_)
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print(str(time.time() - t0) + " seconds wall time.")

t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: kNN", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
testaccuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test accuracy for tuned params is " + str(testaccuracy))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()
