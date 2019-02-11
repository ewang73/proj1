from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
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
traindata = pd.read_csv("SFCrimeBinaryUndersampled.csv", sep=",", low_memory= False)
X = traindata.values[:5000, 1:]
Y = traindata.values[:5000, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)

# Find best hyperparameters
t0= time.time()
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
degrees = [1,2,3,4,5]
param_grid = [{'kernel': ['linear', 'rbf'], 'C': Cs, 'gamma': gammas},
              {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}]

clf = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=10)

clf.fit(X_train, Y_train)
print(clf.best_params_)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print('----------------')
print(clf.cv_results_)
print(str(time.time() - t0) + " seconds wall time.")


# visualize across different parameters (change from knn)
lin_mean_test_score = [0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 ,
       0.5735 , 0.58925, 0.58925, 0.58925, 0.58925, 0.58475, 0.58475,
       0.58475, 0.58475, 0.58275, 0.58275, 0.58275, 0.58275]
rbf_mean_test_score = [0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 ,
       0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 ,
       0.59925, 0.56475, 0.5735 , 0.58975, 0.58425, 0.5585 ]
poly2_mean_test_score = [0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 , 0.5735 ,
       0.586  , 0.5735 , 0.5735 , 0.5735 , 0.591  , 0.5735 , 0.5735 ,
       0.586  , 0.5705 , 0.5735 , 0.5735 , 0.59175, 0.5695 ]
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
kernels = ['linear', 'rbf', 'poly2']
data = [lin_mean_test_score, rbf_mean_test_score, poly2_mean_test_score]

for dataSet in range(3):
    plt.figure(figsize=(8,6))
    plt.title(f'SVM: C vs Accuracy, kernel={kernels[dataSet]}')
    for i, gamma in enumerate(gammas): #using linear kernel
        C_test_scores = [data[dataSet][i + j*4] for j in range(5)]
        plt.semilogx(Cs, C_test_scores, label=f'test scores gamma={gamma}')
    plt.legend()
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.show()

for dataSet in range(3):
    plt.figure(figsize=(8,6))
    plt.title(f'SVM: Gamma vs Accuracy, kernel={kernels[dataSet]}')
    for i, C in enumerate(Cs): #using linear kernel
        gamma_test_scores = [data[dataSet][i*4 + j] for j in range(4)]
        plt.semilogx(gammas, gamma_test_scores, label=f'test scores C={C}')
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.show()

# prob score
# t1 = time.time()

# clf = SVC(kernel='linear', C=1, gamma=.001, probability=True)
# clf = clf.fit(X_train, Y_train)

# Y_probs = clf.predict_proba(X_train)
# # print(Y_probs[:100])

# YpredictionArray = []
# for prob in Y_probs:
#     if prob[1] > 108374/878049:
#         YpredictionArray.append(1)
#     else:
#         YpredictionArray.append(0)
# train_acc = accuracy_score(Y_train, np.array(YpredictionArray))
# print(train_acc)
# print(sum(YpredictionArray))

# print(str(time.time() - t1) + " seconds wall time.")


# plot learning curve best params
t1 = time.time()
clf = SVC(kernel='poly', degree=2, C=10, gamma=.1)
clf = clf.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: SVM - poly2", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

t2 = time.time()
clf = SVC(kernel='rbf', C=1, gamma=.1)
clf = clf.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: SVM - rbf", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t2) + " seconds wall time.")
plt.show()

t3 = time.time()
clf = SVC(kernel='linear', C=.1, gamma=.001)
clf = clf.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: SVM - linear", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t3) + " seconds wall time.")
plt.show()

# ---------------------------------------------------------------------------

#Plotting some kernels and stuff: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

for i in range(47):
    # find farthest combination
    X = traindata.values[:5000, [0,i]]

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
      # SVM regularization parameter
    models = (SVC(kernel='linear', C=.1, gamma=.001),
              LinearSVC(C=1),
              SVC(kernel='rbf', gamma=0.1, C=1),
              SVC(kernel='poly', degree=2, C=10, gamma=.01))
    models = (clf.fit(X, Y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 2) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(f'Feature 0')
        ax.set_ylabel(f'Feature {i}')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()
