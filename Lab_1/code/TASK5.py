import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import metrics

income_data = pd.read_csv('income_evaluation.csv')

# Null Values
n = pd.DataFrame(income_data.isnull().sum().sort_values(ascending=False))
n.columns = ['Null Count']
n.index.name = 'Feature'
print(n)


# Finding correlation with the target class
income_data[' income'] =income_data[' income'].map( {' <=50K': 1, ' >50K': 0} ).astype(int)
features = income_data

corr = features.corr()
print(corr[' income'].sort_values(ascending=False), '\n')
#print(corr['income'])

income_data=income_data.drop(columns=[' fnlwgt', ' workclass', ' education', ' marital-status',
       ' occupation', ' relationship', ' race', ' sex', ' native-country'],axis=1)
print(income_data.columns)

# Splitting the data set
rng = RandomState()

train = income_data.sample(frac=0.7, random_state=rng)
test = income_data.loc[~income_data.index.isin(train.index)]
X_train = train.drop(" income",axis=1)
Y_train = train[" income"]
X_test = test.drop(" income",axis=1)
Y_test = test[" income"]

##KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:",acc_knn)

##SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("svm accuracy is:", acc_svc)

##Naive Bayes
model = GaussianNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
expected = Y_train
predicted = model.predict(X_train)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))