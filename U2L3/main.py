#Implement Linear Regression in Python using Numpy
# Import Statements
import numpy as np
from matplotlib import pyplot as plt
# =================================================================
# Complete the code for linear regression
# Code should be vectorized when complete
# ================================================================
# Create test data and split into features and labels
def createData():
  data = np.array([[1,0,0],
  [1,1,0.5],
  [1,2,1],
  [1,3,1.5]]).astype(float)

  x = data[:, 0:2]
  y = data[:,2]
  return x,y

# *** function to create an array of costs for each example
def calcCost(X,W,Y):
  p = np.dot(X,W)
  cost = np.sum((p-Y)**2)
  cost /= len(Y)
  
  return cost

# *** Code that calculates 1/m * d/dw (cost function)
def calcGradient(X,Y,W):
  p = np.dot(X,W)
  error = p - Y
  gradient = np.dot(error, X)
  gradient /= len(Y)
  return gradient

X,Y = createData()

# Set initial weights
W = [0, 0]
# set learning rate - the list is if we want to try multiple LR's
lrList = [.3, .01]
# We are only using one of them today
lr = lrList[0]
#set up the cost array for graphing
costArray = []
costArray.append(calcCost(X, W, Y))
#initalize while loop flags
finished = False
count = 0
while (not finished and count <10):
  gradient = calcGradient(X,Y,W)
  print ("gradient", gradient)
  #*** update weights --------------------------------
  W -= lr*gradient
  print("weights: ", W)
  costArray.append(calcCost(X, W, Y))
  lengthOfGradientVector = np.linalg.norm(gradient)
  if (lengthOfGradientVector < .00001):
    finished=True
  count+=1
####################### graphs #############################
# Create figure objects
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8]) #[left, bottom, width, height]
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")
ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()
plt.show()