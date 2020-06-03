# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:33:53 2020

@author: Prathamesh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


df=pd.read_csv('new_data.csv')

mse=[]
score=[]
df_y=df['adr']

def iter(k):
    fs=SelectKBest(f_regression,k)
    print(fs)
    df_x = fs.fit_transform(df,df['adr'])

    df_x_train=df_x[:100000]
    df_x_test=df_x[100000:]

    df_y_train=df_y[:100000]
    df_y_test=df_y[100000:]

    regr=linear_model.LinearRegression()
    regr.fit(df_x_train,df_y_train)

    df_y_pred=regr.predict(df_x_test)
    #create_polynomial_regression_model(4)
    s=regr.score(df_x_test,df_y_test)
    print("Score: ",s)
    #print('Coefficients:\n',regr.coef_)
    m=mean_squared_error(df_y_test,df_y_pred)
    print('Mean Squared Error:%2.f'%m)
    score.append(s)
    mse.append(m)
    #plt.show()

for i in range(1,31):
    iter(i)
plt.plot(score)
plt.show()
plt.plot(mse)
plt.plot()

# Using the top 18 attributes selected by SelectKBest gives the highest accuracy.
