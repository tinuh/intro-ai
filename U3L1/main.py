import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = np.array([ [5,10,1], [5,6,0], [-4,0,1], [-5,-6,0], [4,8,1], [0,0,0], [-6,4,1], [4,0,0], [2,-5,0], [4,-5,0], [-4,-3,0] ]).astype(float)

# split data into features and labels
X = data[:,0:2]
Y = data[:,2]

# add a column of ones to X
X = np.c_[np.ones(X.shape[0]), X]

# set initial weights
W = [0, 0, 0]
LR = 0.2

finished = False
count = 0

costArray = []

# iterations
while (not finished and count < 100000):
  # calculate gradient
  p = 1 / (1 + np.exp(-np.dot(X, W)))
  error = p - Y
  gradient = np.dot(error, X)
  gradient /= len(Y)
  #print(f'Gradient: {gradient}')

  # update weights
  W -= LR * gradient
  # print(f'Weights: {W}')

  # calculate cost (linear regression)
  # p = np.dot(X, W)
  # cost = np.sum((p - Y) ** 2)
  # cost /= len(Y)
  # costArray.append(cost)

  # calculate cost (logistic regression)
  cost = 0
  for i in range(len(Y)):
    p = 1 / (1 + np.exp(-np.dot(W, X[i])))
    cost -= Y[i] * np.log(p) + (1 - Y[i]) * np.log(1 - p)
  cost /= -len(Y)
  costArray.append(cost)

  # check if gradient is small enough
  lengthOfGradientVector = np.linalg.norm(gradient)
  if (lengthOfGradientVector < 0.0001):
    finished = True
  count += 1

print(f'Final weights: {W}')
print(f'Final cost: {costArray[-1]}')
print(f'Iterations: {count}')

# plot costs over time
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) #[left, bottom, width, height]
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")
ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()
plt.show()