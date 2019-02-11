from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
import graphviz
from sklearn.metrics import precision_recall_fscore_support

##### CRIME DATASET #####
# Preparing data
traindata = pd.read_csv("SFCrimeClean.csv", sep=",", low_memory= False)
X = traindata.values[:, :5]
Y = traindata.values[:, 5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)
SAMPLE_PROPORTION = 108374/878049 #hardcoded sample proportion

# Initial testing, default params
t0 = time.time()
clf = tree.DecisionTreeClassifier(criterion = "gini", splitter='best', min_samples_leaf = 1, max_depth=None)
clf = clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for default params is " + str(trainaccuracy))
print("-------------------------------------")
testaccuracy = precision_recall_fscore_support(Y_test, clf.predict(X_test))
print(testaccuracy)
print(str(time.time() - t0) + " seconds wall time.")

with open("crime.DT.graphs/initialTree.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)

# Find best hyperparameters, plot_learning_curve
t1 = time.time()
param_grid = {'max_depth': [i for i in range(2, 20)], 'max_features': [1, 2, 3, 4, 5]}
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=3, refit=True)
clf.fit(X_train, Y_train)
print(clf.best_params_)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(str(time.time() - t1) + " seconds wall time.")

t1o5 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Decision Trees", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t1o5) + " seconds wall time.")
plt.show()

with open("crime.DT.graphs/prunedTree.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)

# Training trees of different max_depths
max_depth = list(range(2, 20))
train_mse = [0] * len(max_depth)
train_err = [0] * len(max_depth)
test_mse = [0] * len(max_depth)
test_err = [0] * len(max_depth)

t2 = time.time()
for i, d in enumerate(max_depth):
    print('learning a decision tree with max_depth=' + str(d))
    clf = tree.DecisionTreeClassifier(max_depth=d)
    X_cvtrain, X_val, Y_cvtrain, Y_val = train_test_split(X_train, Y_train, test_size=.2, random_state= 20)
    clf = clf.fit(X_cvtrain, Y_cvtrain)

    train_mse[i] = mean_squared_error(Y_cvtrain, clf.predict(X_cvtrain))
    Y_probs = clf.predict_proba(X_cvtrain)
    YpredictionArray = []
    for prob in Y_probs:
        if prob[1] > SAMPLE_PROPORTION:
            YpredictionArray.append(1)
        else:
            YpredictionArray.append(-1)
    # train_mse[i] = mean_squared_error(Y_cvtrain, np.array(YpredictionArray))
    train_err[i] = 1 - accuracy_score(Y_cvtrain, np.array(YpredictionArray))

    test_mse[i] = mean_squared_error(Y_val, clf.predict(X_val))
    Y_probs = clf.predict_proba(X_val)
    YpredictionArray = []
    for prob in Y_probs:
        if prob[1] > SAMPLE_PROPORTION:
            YpredictionArray.append(1)
        else:
            YpredictionArray.append(-1)
    # test_mse[i] = mean_squared_error(Y_val, np.array(YpredictionArray))
    test_err[i] = 1 - accuracy_score(Y_val, np.array(YpredictionArray))

    # print('train_mse: ' + str(train_mse[i]))
    # print('train_err: ' + str(train_err[i]))
    # print('test_mse: ' + str(test_mse[i]))
    # print('test_err: ' + str(test_err[i]))
    # print('---')

print(str(time.time() - t2) + " seconds wall time.")

# Plot results, sample proportion error, classification mse
print('plotting results')
plt.figure()
plt.title('Decision Trees: Performance x Max Depth')
plt.plot(max_depth, test_mse, '-', label='validation mse')
plt.plot(max_depth, train_mse, '-', label='train mse')
plt.plot(max_depth, test_err, '-', label='validation error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.show()
### ---

# Training trees of different training set sizes (best hyperparams)
train_size = len(X_train)
offsets = list(range(int(0.1 * train_size), train_size, int(0.05 * train_size)))
MAX_DEPTH = 6
MAX_FEATURES = 5
train_mse = [0] * len(offsets)
train_err = [0] * len(offsets)
test_mse = [0] * len(offsets)
test_err = [0] * len(offsets)

print('training_set_max_size:', train_size, '\n')

t3 = time.time()
for i, o in enumerate(offsets):
    print('learning a decision tree with training_set_size=' + str(o))
    clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH, max_features=MAX_FEATURES)
    clf = clf.fit(X_train[:o], Y_train[:o])

    # train_mse[i] = mean_squared_error(Y_train[:o], clf.predict(X_train[:o]))
    Y_probs = clf.predict_proba(X_train[:o])
    YpredictionArray = []
    for prob in Y_probs:
        if prob[1] > SAMPLE_PROPORTION:
            YpredictionArray.append(1)
        else:
            YpredictionArray.append(-1)
    train_mse[i] = mean_squared_error(Y_train[:o], np.array(YpredictionArray))
    train_err[i] = 1 - accuracy_score(Y_train[:o], np.array(YpredictionArray))

    # test_mse[i] = mean_squared_error(Y_test[:o], clf.predict(X_test[:o]))
    Y_probs = clf.predict_proba(X_test[:o])
    YpredictionArray = []
    for prob in Y_probs:
        if prob[1] > SAMPLE_PROPORTION:
            YpredictionArray.append(1)
        else:
            YpredictionArray.append(-1)
    test_mse[i] = mean_squared_error(Y_test[:o], np.array(YpredictionArray))
    test_err[i] = 1 - accuracy_score(Y_test[:o], np.array(YpredictionArray))

    # print('train_mse: ' + str(train_mse[i]))
    # print('train_err: ' + str(train_err[i]))
    # print('test_mse: ' + str(test_mse[i]))
    # print('test_err: ' + str(test_err[i]))
    # print('---')

    # train_err[i] = 1 - accuracy_score(Y_train[:o], clf.predict(X_train[:o]))
    # test_err[i] = 1 - accuracy_score(Y_test[:o], clf.predict(X_test[:o]))

print(str(time.time() - t3) + " seconds wall time.")

# Plot results
print('plotting results')
plt.figure()
plt.title('Decision Trees: Performance x Training Set Size')
# plt.plot(offsets, test_mse, '-', label='test mse')
# plt.plot(offsets, train_mse, '-', label='train mse')
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.show()
### ---

### Undersampled training
# Preparing data
traindata = pd.read_csv("SFCrimeUndersampled.csv", sep=",", low_memory= False)
X = traindata.values[:, :5]
Y = traindata.values[:, 5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)
SAMPLE_PROPORTION = 108374/250000 #hardcoded sample proportion

#Initial Testing, default params
t4 = time.time()
clf = tree.DecisionTreeClassifier(criterion = "gini", splitter='best', min_samples_leaf = 1, max_depth=None)
clf = clf.fit(X_train, Y_train)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for default params is " + str(trainaccuracy))
print(str(time.time() - t4) + " seconds wall time.")
print("-------------------------------------")
testaccuracy = precision_recall_fscore_support(Y_test, clf.predict(X_test))
print(testaccuracy)

with open("crime.DT.graphs/initialTreeUndersampled.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)

# Find best hyperparameters, plot_learning_curve
t5 = time.time()
param_grid = {'max_depth': [i for i in range(2, 20)], 'max_features': [1, 2, 3, 4, 5]}
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=3, refit=True)
clf.fit(X_train, Y_train)
print(clf.best_params_)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(str(time.time() - t5) + " seconds wall time.")

t5o5 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: Decision Trees", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t5o5) + " seconds wall time.")
plt.show()

with open("crime.DT.graphs/prunedTreeUndersampled.txt", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f)

# Training trees of different max_depths
max_depth = list(range(2, 20))
train_mse = [0] * len(max_depth)
train_err = [0] * len(max_depth)
test_mse = [0] * len(max_depth)
test_err = [0] * len(max_depth)

t6 = time.time()
for i, d in enumerate(max_depth):
    print('learning a decision tree with max_depth=' + str(d))
    clf = tree.DecisionTreeClassifier(max_depth=d)
    X_cvtrain, X_val, Y_cvtrain, Y_val = train_test_split(X_train, Y_train, test_size=.2, random_state= 20)
    clf = clf.fit(X_cvtrain, Y_cvtrain)

    train_mse[i] = mean_squared_error(Y_cvtrain, clf.predict(X_cvtrain))
    train_err[i] = 1 - accuracy_score(Y_cvtrain, clf.predict(X_cvtrain))

    test_mse[i] = mean_squared_error(Y_val, clf.predict(X_val))
    test_err[i] = 1 - accuracy_score(Y_val, clf.predict(X_val))

print(str(time.time() - t6) + " seconds wall time.")

# Plot results, sample proportion error, classification mse
print('plotting results')
plt.figure()
plt.title('Decision Trees: Performance x Max Depth')
# plt.plot(max_depth, test_mse, '-', label='test mse')
# plt.plot(max_depth, train_mse, '-', label='train mse')
plt.plot(max_depth, test_err, '-', label='test error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.show()
### ---
