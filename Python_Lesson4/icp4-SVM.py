import pandas as pd

glassdata = pd.read_csv("glass.csv")

# Data preprocessing


X = glassdata.drop('Type', axis=1)

Y = glassdata['Type']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=200)

from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf', gamma='auto')

svclassifier.fit(X_train, Y_train)

Y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))