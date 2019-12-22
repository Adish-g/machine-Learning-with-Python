import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline


# Download the data
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

# Reading the data in
a=pd.read_csv("FuelConsumption.csv")
a.head(n=10)

# Data Exploration
a.describe()

b=a[['ENGINESIZE',"MODELYEAR","CYLINDERS",'CO2EMISSIONS']]
b.head()

# Plot the data
visul=b[["MODELYEAR","CYLINDERS",'CO2EMISSIONS','ENGINESIZE']]
visul.hist()
plt.show()

# Plot based on the Engine_size Vs Co2_Emission
plt.scatter(b.ENGINESIZE, b.CO2EMISSIONS, color="pink")
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 EMISSION")
plt.show()




         # .        Creating train and test dataset
  
  
# Dividing the data .   80-20% .  ( 80- train, 20- test)
z=np.random.rand(len(a))<0.80             # . z contains boolean values
train=b[z]                # training data set contains only true.
test=b[~z]                # testing data set contains only false

np.count_nonzero(b)             # Total number of data present in training set
len(z)-np.count_nonzero(b)     # Total number of data present in testing set



# Training set & Modeling
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()   

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

#Plot output
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")



# Testing
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
  
  
  
  
