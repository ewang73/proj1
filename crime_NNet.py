from sklearn.neural_network import MLPClassifier
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
X = traindata.values[:, 1:]
Y = traindata.values[:, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state= 20)

# Find best hyperparameters, plot_learning_curve
t0= time.time()
param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(10,),(20,),(30,),(40,),(50,)], 'learning_rate': ['constant', 'invscaling']}
clf = GridSearchCV(MLPClassifier(solver='sgd', max_iter=1000), param_grid, cv=3, refit=True, verbose=10)
clf.fit(X_train, Y_train)
print(clf.best_params_)
trainaccuracy = accuracy_score(Y_train, clf.predict(X_train))*100
print("The training accuracy for tuned params is " + str(trainaccuracy))
print(clf.best_score_)
print('----------------')
print(clf.cv_results_)
print(str(time.time() - t0) + " seconds wall time.")

# learning curve of best hyperparameters, crossvalidated
t1 = time.time()
clf = clf.best_estimator_.fit(X_train, Y_train)
skplt.estimators.plot_learning_curve(clf, X, Y, title = "Learning Curve: kNN", cv=5, train_sizes=np.linspace(.1, 1.0, 20))
print(str(time.time() - t1) + " seconds wall time.")
plt.show()

# plotting parameters
mean_test_score = array([0.596805, 0.561395, 0.597095, 0.556885, 0.596985, 0.55985 ,
       0.597045, 0.571075, 0.597005, 0.568295, 0.595295, 0.56655 ,
       0.59578 , 0.56655 , 0.59569 , 0.56655 , 0.595965, 0.56655 ,
       0.59615 , 0.56655 , 0.607315, 0.551135, 0.611545, 0.555615,
       0.604435, 0.563005, 0.608005, 0.56911 , 0.603335, 0.562515,
       0.61361 , 0.55814 , 0.613915, 0.55331 , 0.61379 , 0.55242 ,
       0.61306 , 0.559475, 0.613255, 0.56325 ])

mean_train_score = array([0.5972925 , 0.56231252, 0.597415  , 0.557045  , 0.59763   ,
       0.56140248, 0.5976075 , 0.5701675 , 0.5975325 , 0.56867748,
       0.595815  , 0.56655   , 0.5959475 , 0.56655   , 0.5963475 ,
       0.56655   , 0.59618   , 0.56655   , 0.5968475 , 0.56655   ,
       0.60864748, 0.5521875 , 0.613065  , 0.55619752, 0.60511498,
       0.56417999, 0.60905999, 0.56968749, 0.60527001, 0.56307999,
       0.6142075 , 0.55800249, 0.6148275 , 0.55289248, 0.6149175 ,
       0.55291001, 0.6149875 , 0.5602275 , 0.61478   , 0.563715  ])


activationArray = ['identity', 'logistic', 'tanh', 'relu']
numNodes = [10,20,30,40,50]
learning_rate = ['constant', 'invscaling']

plt.figure(figsize=(8,6))
plt.title(f'Neural Nets: Number of Nodes vs Accuracy, learning=constant')
for i, activation in enumerate(activationArray): #using constant learning rate
    constant_test_scores = [mean_test_score[i*10 + j*2] for j in range(5)]
    plt.plot(numNodes, constant_test_scores, label=f'test scores {activation}')
plt.legend()
plt.xlabel('Nodes')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(8,6))
plt.title(f'Neural Nets: Number of Nodes vs Accuracy, activation={activation}')
for i, activation in enumerate(activationArray): #using both learning rates
    constant_test_scores = [mean_test_score[i*10 + j*2] for j in range(5)]
    plt.plot(numNodes, constant_test_scores, '0.5')
    invscaling_test_scores = [mean_test_score[i*10 + j*2 + 1] for j in range(5)]
    plt.plot(numNodes, invscaling_test_scores, label=f'test scores {activation} - invscaling')
plt.legend()
plt.xlabel('Nodes')
plt.ylabel('Accuracy')
plt.show()
