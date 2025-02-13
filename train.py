import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linearRegression import LinearRegression

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

reg = LinearRegression(lr=0.01, n_iters=1000)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print("Mean Squared Error = ", mse)

# plot the data and the fitted line
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, label="Predictions")
plt.show()


