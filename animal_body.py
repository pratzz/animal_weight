from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataframe = pd.read_fwf("brain_body.txt")
brain_values = dataframe[['Brain']]
body_values = dataframe[['Body']]


reg = linear_model.LinearRegression()
reg.fit (brain_values,body_values)

#visualize results
plt.scatter(brain_values, body_values)
plt.plot(brain_values, reg.predict(brain_values))
plt.show()