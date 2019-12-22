import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as py
%matplotlib inline


# Reading the data in
a=pd.read_csv("FuelConsumption.csv")
a.head(n=10)

# Data Exploration
a.describe()

b=a[['ENGINESIZE',"MODELYEAR","CYLINDERS",'CO2EMISSIONS','FUELCONSUMPTION_CITY',"FUELTYPE"]]
b.head()

# Plot the data
visul=b[["MODELYEAR","CYLINDERS",'CO2EMISSIONS','ENGINESIZE','FUELCONSUMPTION_CITY',"FUELTYPE"]]
visul.hist()
plt.show()

# Plot based on the Engine_size Vs Co2_Emission
plt.scatter(b.ENGINESIZE, b.CO2EMISSIONS, color="pink")
plt.xlabel("ENGINE SIZE")
plt.ylabel("CO2 EMISSION")
plt.show()



                      # Traing and Testing
                      
z=np.random.rand(len(b))<0.75
train=b[z]
test=b[~z]

np.count_nonzero(z)
len(b)-np.count_nonzero(z)


# Train
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.xlabel("Engine Size")
plt.ylabel("Co2 Smission")
plt.show()


from sklearn import linear_model
reg=linear_model.LinearRegression()
x=np.asanyarray(train[['ENGINESIZE',"CYLINDERS",'FUELCONSUMPTION_CITY']])
y=np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(x,y)
print(" Coefficient :",reg.coef_)


y_hat=reg.predict(test[['ENGINESIZE',"CYLINDERS",'FUELCONSUMPTION_CITY']])
x=np.asanyarray(test[['ENGINESIZE',"CYLINDERS",'FUELCONSUMPTION_CITY']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print("MEAN SQUARE ERROR :",np.mean((y_hat-y)**2))
print("VARIANCE SCORE :",reg.score(x,y))







