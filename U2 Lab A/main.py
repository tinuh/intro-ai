import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read in the data
df = pd.read_csv('Loudoun_Housing_Data.csv')
print(df.head())

# standardize the square feet column
df['Square Feet'] = (df['Square Feet'] - df['Square Feet'].mean()) / df['Square Feet'].std()

# drop na values
df = df.dropna()

# create a scatter plot of square feet vs. price
plt.scatter(df['Square Feet'], df['Price'], s=8)
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Square Feet vs. Price')
plt.show()

# create a scatter plot of bathrooms vs. price
plt.scatter(df['Bathrooms'], df['Price'], s=8)
plt.xlabel('Bathrooms')
plt.ylabel('Price')
plt.title('Bathrooms vs. Price')
plt.show()

#create a scatter plot of bedrooms vs. price
plt.scatter(df['Bedrooms'], df['Price'], s=8)
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Bedrooms vs. Price')
plt.show()

# split data into features and labels
# features = square feet, bedrooms, labels = price
X = np.array(df['Square Feet'])
Y = np.array(df['Price'])

# add a column of ones to X
X = np.c_[np.ones(X.shape[0]), X]

# set initial weights
W = [0, 0]
LR = 0.05
df_cost = pd.DataFrame(columns=['cost'])

finished = False
count = 0

costArray = []

# iterations
while (not finished and count < 100):
  # calculate gradient
  p = np.dot(X, W)
  error = p - Y
  gradient = np.dot(error, X)
  gradient /= len(Y)
  print(f'Gradient: {gradient}')

  # update weights
  W -= LR * gradient
  print(f'Weights: {W}')

  # calculate cost
  p = np.dot(X, W)
  cost = np.sum((p - Y) ** 2)
  cost /= len(Y)
  costArray.append(cost)

  # check if gradient is small enough
  lengthOfGradientVector = np.linalg.norm(gradient)
  if (lengthOfGradientVector < 0.00001):
    finished = True
  count += 1

# plot costs over time
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) #[left, bottom, width, height]
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")
ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()
plt.show()

# create a scatter plot of square feet vs. price
plt.scatter(df['Square Feet'], df['Price'], s=8)
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Square Feet vs. Price')
plt.plot(df['Square Feet'], np.dot(X, W), color='red')

# plot the line using the weights array
plt.plot(df['Square Feet'], np.dot(X, W), color='red')
plt.show()

# predict the price of a house with 2000 square feet
print(f'Predicted price of a house with 2000 square feet: {np.dot([1, 2000], W)}')
print(f'Final Cost: {costArray[-1]}')