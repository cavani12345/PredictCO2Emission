import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('FuelConsumption.csv')
newdata = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

x = newdata.iloc[:,:-1]
y = newdata.iloc[:,-1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

pickle.dump(regressor,open('model.pkl','wb'))
