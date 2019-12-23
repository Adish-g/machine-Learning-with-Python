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

np.count_nonzero(z)             # Total number of data present in training set
len(z)-np.count_nonzero(z)     # Total number of data present in testing set





# NOTE: PolynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set. That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, lets say the original feature set has only one feature, ENGINESIZE. Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])

test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])

poly=PolynomialFeatures(degree=2)
train_x_poly=poly.fit_transform(train_x)
train_x_poly

reg=linear_model.LinearRegression()
train_y_lin=reg.fit(train_x_poly,train_y)
print("Coefficient :",reg.coef_)
print("Intercept   :" , reg.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )





