import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def create_polynomial_regression_model(degree):
  "Creates a polynomial regression model for the given degree"

  poly_features = PolynomialFeatures(degree=degree)

  # transforms the existing features to higher degree features.
  X_train_poly = poly_features.fit_transform(df_x_train)

  # fit the transformed features to Linear Regression
  poly_model = LinearRegression()
  poly_model.fit(X_train_poly, df_y_train)

  # predicting on training data-set
  y_train_predicted = poly_model.predict(X_train_poly)

  # predicting on test data-set
  y_test_predict = poly_model.predict(poly_features.fit_transform(df_x_test))

  # evaluating the model on training dataset
  rmse_train = np.sqrt(mean_squared_error(df_y_train, y_train_predicted))
  r2_train = r2_score(df_y_train, y_train_predicted)

  # evaluating the model on test dataset
  rmse_test = np.sqrt(mean_squared_error(df_y_test, y_test_predict))
  r2_test = r2_score(df_y_test, y_test_predict)

  print("Degree: ",degree)
  print("The model performance for the training set")
  print("-------------------------------------------")
  print("RMSE of training set is {}".format(rmse_train))
  print("R2 score of training set is {}".format(r2_train))

  print("The model performance for the test set")
  print("-------------------------------------------")
  print("RMSE of test set is {}".format(rmse_test))
  print("R2 score of test set is {}".format(r2_test))
  print("\n")

  rmse.append(rmse_test)
  r2.append(r2_test)

df=pd.read_csv('new_data.csv')
rmse=[]
r2=[]
df_y=df['adr']

# Since top 18 attributes gives the best accuracy in Linear Regression model, we will use only these atrributes for further analysis.
fs=SelectKBest(f_regression,18)
print(fs)
df_x = fs.fit_transform(df,df['adr'])

df_x_train=df_x[:100000]
df_x_test=df_x[100000:]

df_y_train=df_y[:100000]
df_y_test=df_y[100000:]

for degree in range(1,4):
    create_polynomial_regression_model(degree)

plt.plot(r2)
plt.show()
plt.plot(rmse)
plt.plot()
