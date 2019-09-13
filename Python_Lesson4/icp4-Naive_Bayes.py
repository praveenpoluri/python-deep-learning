



from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

import pandas as pd

# Loading the data set using pandas


glass = pd.read_csv("glass.csv")

# Preprocessing data


X = glass.drop('Type', axis=1)

Y = glass['Type']

# splitting data into training data and testing data


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=20)

# creating and Training the classifier


model = GaussianNB()

model.fit(X_train, Y_train)

# Prediction


Y_pred = model.predict(X_test)

# Evaluation


print("accuracy score:", metrics.accuracy_score(Y_test, Y_pred))

