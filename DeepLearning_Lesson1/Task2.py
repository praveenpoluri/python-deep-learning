import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

# Load the train DataFrames using pandas
dataset = pd.read_csv('Breas Cancer.csv')




import numpy as np

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv("Breas Cancer.csv")
# print(dataset)


from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()
X = dataset.iloc[:, 2:32].values
dataset['diagnosis'].replace('M', 1,inplace=True)
dataset['diagnosis'].replace('B', 0,inplace=True)
y = dataset.iloc[:, 1].values



dataset = dataset.values



X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:32], dataset[:,1],
                                                    test_size=0.25, random_state=87)
from sklearn.preprocessing import StandardScaler

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(15, input_dim=30, activation='relu'))
my_first_nn.add(Dense(50, activation='relu'))
my_first_nn.add(Dense(100, activation='relu'))

my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))