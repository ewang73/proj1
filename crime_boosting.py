from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scikitplot as skplt
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

##### CRIME DATASET #####
#Read data in using pandas
traindata = pd.read_csv("SFCrimeUndersampled.csv", sep=",", low_memory= False)
X = traindata.values[:, :5]
Y = traindata.values[:, 5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)

# Find best hyperparameters, plot_learning_curve
#we know the hyperparameters for DT of the undersampled set
t0= time.time()
dt = DecisionTreeClassifier(max_depth=10, max_features=5)
param_grid = {'n_estimators': [50, 100, 150, 200, 250, 2000], 'learning_rate': [1., .5, .1]}
clf = GridSearchCV(AdaBoostClassifier(dt), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
print(clf.best_params_)
print(clf.best_score_)
print('----------------')
print(clf.cv_results_)
print(str(time.time() - t0) + " seconds wall time.")

# plotting parameters
test_scores = [[0.594085, 0.593935, 0.594155, 0.59424 , 0.594095, 0.594535],
    [0.59632 , 0.59452 , 0.594225, 0.594705, 0.594485, 0.59473],
    [0.60826 , 0.60417 , 0.60087 , 0.59829 , 0.59695 , 0.594505]]
train_scores = [[0.6454425, 0.6454425, 0.6454425, 0.6454425, 0.6454425, 0.6454425],
    [0.6452675, 0.6454425, 0.6454425, 0.6454425, 0.6454425, 0.6454425],
    [0.6371625, 0.6421175, 0.6438575, 0.6448725, 0.645175 , 0.6454425]]

lrArray = [1.,.5,.1]
n_estArray = [50,100,150,200,250]

plt.figure()
for i, lr in enumerate(lrArray):
    # plt.plot(n_estArray, train_scores[i], label='training scores')
    plt.plot(n_estArray, test_scores[i][:5], label=f'test scores lr={lr}')
plt.title(f'Boosting Decision Trees: Max Estimators vs Accuracy')
plt.legend()
plt.xlabel('Estimators')
plt.ylabel('Accuracy')
plt.show()

n_estArray.append(2000)
plt.figure()
for i, ne in enumerate(n_estArray):
    # plt.plot(lrArray, [train_scores[0][i],train_scores[1][i],train_scores[2][i]], label='training scores')
    plt.plot(lrArray, [test_scores[0][i],test_scores[1][i],test_scores[2][i]], label=f'test scores ne={ne}')
plt.title(f'Boosting Decision Trees: Learning Rate vs Accuracy')
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.show()

# learning curve of best hyperparameters, crossvalidated
t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: kNN", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

# ---------------------------------------

"""
#estimators vs error rate plot (copied from http://scikit-learn.org/stable/auto
#_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemb
#le-plot-adaboost-hastie-10-2-py

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, Y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, Y_test)
n_estimators = 2000

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, Y_train)
dt_err = 1.0 - dt.score(X_test, Y_test)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=1,
    n_estimators= n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, Y_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

#graphic ada error
ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, Y_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, Y_train)

ax.plot(np.arange(n_estimators) + 1, ada_real_err,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, 0.0015))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()
"""
