from sklearn.neighbors import KNeighborsClassifier
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
traindata = pd.read_csv("SFCrimeBinary.csv", sep=",", low_memory= False)
X = traindata.values[:50000, 1:]
Y = traindata.values[:50000, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)

# Find best hyperparameters, plot_learning_curve
t0= time.time()
param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7]} # 'metric': ['jaccard', 'matching', 'dice']
clf = GridSearchCV(KNeighborsClassifier(weights='distance', metric='jaccard'), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
print(clf.best_params_)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print('----------------')
print(clf.cv_results_)
print(str(time.time() - t0) + " seconds wall time.")

# visualize the best n
test_scores = [0.860725, 0.843925, 0.864025, 0.859325, 0.865475, 0.864525]
train_scores = [0.8851375 , 0.87727501, 0.88964999, 0.88827498, 0.89098749, 0.89074999]
kArray = [2, 3, 4, 5, 6, 7]
plt.figure()
plt.title('Decision Trees: Performance x Max Depth')
plt.title("K vs Accuracy")
plt.plot(kArray, train_scores, label='training scores')
plt.plot(kArray, test_scores, label='test scores')
plt.legend()
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

# learning curve of best hyperparameters, crossvalidated
t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: kNN", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

"""
# Best hyperparameters - Test Set - probably don't need
t2 = time.time()
K_BEST = 5
clf = KNeighborsClassifier(n_neighbors = 5, weights = "distance", metric="jaccard")

trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for this is " + str(trainaccuracy))

accuracy = accuracy_score(Y_test, clf.predict(X_test))*100
print("The test classification works with " + str(accuracy) + "% accuracy")


# do we need precision and loss????
# #classification precision score, metrics log loss
# from sklearn.metrics import precision_score
# from sklearn.metrics import log_loss

# precision = precision_score(Y_test, Y_prediction, average = "weighted")*100
# loss = log_loss(Y_test, Y_prediction)*100
# print("Precision: " + str(precision))
# print("Loss: " + str(loss))

print(str(time.time() - t2) + " seconds wall time.")
"""

# add manual LC?
