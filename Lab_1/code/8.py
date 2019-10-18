import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#loading data
w_data = load_wine()
winedata= pd.DataFrame(w_data.data, columns=w_data.feature_names)
print(winedata.columns)
winedata['proline'] = w_data.target

# Before EDA


#Data Splitting
A = winedata.drop('proline', axis=1)
B = winedata['proline']
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.2, random_state=9)

#regression model creation and training it before EDA
mdl=LinearRegression().fit(A_train,B_train)

#target prediction
p=mdl.predict(A_test)

#evaluation of model
mean_sq_1 = mean_squared_error(B_test, p)
r2_sq_1= r2_score(B_test,p)
print("MSE:",mean_sq_1)
print("R2_score : ",r2_sq_1)


### After EDA

# null value count display
n = pd.DataFrame(winedata.isnull().sum().sort_values(ascending=False)[:25])
n.columns = ['Null Count']
n.index.name = 'Feature'
print(n)

#null values replaced with mean
winedata= winedata.select_dtypes(include=[np.number]).interpolate().dropna()

#Finding correlation with target class
num_features = winedata.select_dtypes(include=[np.number])
corr = num_features.corr()
print (corr['proline'].sort_values(ascending=False)[1:4], '\n')


#columns dropped which have less correlation
winedata=winedata.drop(columns=['hue', 'color_intensity', 'proanthocyanins', 'flavanoids', 'total_phenols', 'magnesium', 'ash', 'alcohol'],axis=1)
print(winedata.columns)

#Splitting data

X = winedata.drop('proline', axis=1)
y = winedata['proline']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=9)
#creation of regression model and training it
mdl=LinearRegression().fit(X_train,y_train)

#target prediction
predict=mdl.predict(X_test)

#evaluation of model
mean_sq = mean_squared_error(y_test, predict)
r2_sq = r2_score(y_test,predict)
print("Mean squared error :",mean_sq)
print("R2 score : ",r2_sq)