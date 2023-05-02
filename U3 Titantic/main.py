import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Titanic_train.csv", engine="python")

df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
#replace na age values with average age (since z-score it would be 0)
df['Age'] = df['Age'].fillna(0)
df['Sex'] = df['Sex'].replace(['male', 'female'], [0, 1])


data = np.array(df[['Age', 'Pclass', 'Sex', 'Survived']]).astype(float)

#data = np.array([ [5,10,1], [5,6,0], [-4,0,1], [-5,-6,0], [4,8,1], [0,0,0], [-6,4,1], [4,0,0], [2,-5,0], [4,-5,0], [-4,-3,0] ]).astype(float)

# split data into features and labels
X = data[:,0:3]
Y = data[:,3]

# add a column of ones to X
X = np.c_[np.ones(X.shape[0]), X]

# set initial weights
W = [0, 0, 0, 0]
LR = 0.06

finished = False
count = 0

costArray = []

# iterations
while (not finished and count < 10000):
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

# find the accuracy of the model
correct = 0
for i in range(len(Y)):
  p = 1 / (1 + np.exp(-np.dot(X, W)))
  if (p[i] > 0.5 and Y[i] == 1):
    correct += 1
  elif (p[i] < 0.5 and Y[i] == 0):
    correct += 1

print(f'Accuracy: {correct / len(Y) * 100}%')


# plot costs over time
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) #[left, bottom, width, height]
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")
ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()
plt.show()