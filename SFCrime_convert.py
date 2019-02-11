# used to convert base crime data to usable data
# produces four csv files, one full, one undersampled, and two binary

import csv
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 30)

trainDataSet = pd.read_csv("SFCrimeOrig.csv", sep=",", low_memory= False)
# print("Dataset: ", trainDataSet.head())

#encode text data to integers using getDummies or cat.codes
trainDataSet['Category'] = trainDataSet.Category.astype('category')
trainDataSet['Category'] = trainDataSet['Category'].cat.codes
trainDataSet['DayOfWeek'] = trainDataSet.DayOfWeek.astype('category')
trainDataSet['DayOfWeek'] = trainDataSet['DayOfWeek'].cat.codes
trainDataSet['PdDistrict'] = trainDataSet.PdDistrict.astype('category')
trainDataSet['PdDistrict'] = trainDataSet['PdDistrict'].cat.codes
trainDataSet['Resolution'] = trainDataSet.Resolution.astype('category')
trainDataSet['Resolution'] = trainDataSet['Resolution'].cat.codes
# traindata = pd.get_dummies(trainDataSet, columns=['Resolution']) # this will one-hot

# print("Dataset: ", traindata.head())
print(trainDataSet.index)

# code month and day/night from 'Dates'
month = []
night = []
for date in trainDataSet['Dates']:
    month.append(int(date[5:7]))
    hour = int(date[11:13])
    if (hour < 18) and (hour >= 6):
        night.append(-1)
    else:
        night.append(1)
trainDataSet["Month"] = pd.Series(month)
trainDataSet["Night"] = pd.Series(night)

violent = []
# count = 0
for cat in trainDataSet['Category']:
    if cat == 0 or cat == 1 or cat == 10 or cat == 15 or cat == 25 or cat == 28:
        violent.append(1)
        # count += 1
    else:
        violent.append(-1)
trainDataSet["Violent"] = pd.Series(violent)

trainDataSet = trainDataSet.drop(['Descript', 'Address', 'X', 'Y', 'Dates', 'Category'], axis=1)
# print(trainDataSet.head())
# print(count)

trainDataSet.to_csv('SFCrimeClean.csv', index=False)

# shuffle data
trainDataSet = trainDataSet.sample(frac=1, random_state=1).reset_index(drop=True)
# print(trainDataSet.head())

##### encode to binary values - 48 columns (47 binary features, 1 binary label)
traindata = trainDataSet.reindex(columns=['Violent', 'Night', 'Resolution', 'DayOfWeek', 'PdDistrict', 'Month'])
traindata = pd.get_dummies(traindata, columns=['Resolution', 'DayOfWeek', 'PdDistrict', 'Month']) # this will one-hot
traindata = traindata.replace(-1, 0)
print(traindata.head())
traindata.to_csv('SFCrimeBinary.csv', index=False)

##### hardcode undersampling
count = 0
nonviolent = []
for row in trainDataSet.itertuples():
    if row[6] == -1:
        nonviolent.append(row[0])
        count += 1
    if count >= 628049:
        break

undersampledSet = trainDataSet.drop(nonviolent)
undersampledSet = undersampledSet.sample(frac=1, random_state=1).reset_index(drop=True)

# print(undersampledSet[:100])
# print(sum(undersampledSet['Violent']))

undersampledSet.to_csv('SFCrimeUndersampled.csv', index=False)

##### binary set undersampling
binUndersampled = undersampledSet.reindex(columns=['Violent', 'Night', 'Resolution', 'DayOfWeek', 'PdDistrict', 'Month'])
binUndersampled = pd.get_dummies(binUndersampled, columns=['Resolution', 'DayOfWeek', 'PdDistrict', 'Month']) # this will one-hot
binUndersampled = binUndersampled.replace(-1, 0)
print(binUndersampled.head())

binUndersampled.to_csv('SFCrimeBinaryUndersampled.csv', index=False)
