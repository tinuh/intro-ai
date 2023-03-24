import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# read in the data
df = pd.read_csv('cereal.csv')
print(df.head())

print(df['calories'].describe())
print(df['sugars'].describe())
print(df['rating'].describe())

# drop na values
df = df.dropna(subset=['sugars', 'rating'])

# standardize the sugars, calories, vitamins column
df['sugars'] = (df['sugars'] - df['sugars'].mean()) / df['sugars'].std()
df['calories'] = (df['calories'] - df['calories'].mean()) / df['calories'].std()
df['fiber'] = (df['fiber'] - df['fiber'].mean()) / df['fiber'].std()
df['vitamins'] = (df['vitamins'] - df['vitamins'].mean()) / df['vitamins'].std()
df['carbo'] = (df['carbo'] - df['carbo'].mean()) / df['carbo'].std()
df['fat'] = (df['fat'] - df['fat'].mean()) / df['fat'].std()
df['protein'] = (df['protein'] - df['protein'].mean()) / df['protein'].std()
df['cups'] = (df['cups'] - df['cups'].mean()) / df['cups'].std()
df['shelf'] = (df['shelf'] - df['shelf'].mean()) / df['shelf'].std()
df['weight'] = (df['weight'] - df['weight'].mean()) / df['weight'].std()
df['potass'] = (df['potass'] - df['potass'].mean()) / df['potass'].std()
df['sodium'] = (df['sodium'] - df['sodium'].mean()) / df['sodium'].std()

# print R^2 value for sugars vs. rating
print(f'R^2 value for sugars vs. rating: {np.corrcoef(df["sugars"], df["rating"])[0, 1] ** 2}')

# print R^2 value for calories vs. rating
print(f'R^2 value for calories vs. rating: {np.corrcoef(df["calories"], df["rating"])[0, 1] ** 2}')

# print R^2 value for vitamins vs. rating
print(f'R^2 value for vitamins vs. rating: {np.corrcoef(df["vitamins"], df["rating"])[0, 1] ** 2}')

# print R^2 value for fiber vs. rating
print(f'R^2 value for fiber vs. rating: {np.corrcoef(df["fiber"], df["rating"])[0, 1] ** 2}')

# print R^2 value for carbo vs. rating
print(f'R^2 value for carbo vs. rating: {np.corrcoef(df["carbo"], df["rating"])[0, 1] ** 2}')

# print R^2 value for fat vs. rating
print(f'R^2 value for fat vs. rating: {np.corrcoef(df["fat"], df["rating"])[0, 1] ** 2}')

# print R^2 value for protein vs. rating
print(f'R^2 value for protein vs. rating: {np.corrcoef(df["protein"], df["rating"])[0, 1] ** 2}')

# print R^2 value for cups vs. rating
print(f'R^2 value for cups vs. rating: {np.corrcoef(df["cups"], df["rating"])[0, 1] ** 2}')

# print R^2 value for shelf vs. rating
print(f'R^2 value for shelf vs. rating: {np.corrcoef(df["shelf"], df["rating"])[0, 1] ** 2}')

# print R^2 value for weight vs. rating
print(f'R^2 value for weight vs. rating: {np.corrcoef(df["weight"], df["rating"])[0, 1] ** 2}')

# print R^2 value for potass vs. rating
print(f'R^2 value for potass vs. rating: {np.corrcoef(df["potass"], df["rating"])[0, 1] ** 2}')

# print R^2 value for sodium vs. rating
print(f'R^2 value for sodium vs. rating: {np.corrcoef(df["sodium"], df["rating"])[0, 1] ** 2}')

# split data into features and labels
# features = sugars, fat labels = rating
X = np.array(df[['sugars', 'calories', 'fiber', 'sodium', 'protein', 'fat', 'potass']])
Y = np.array(df['rating'])

# add a column of ones to X
X = np.c_[np.ones(X.shape[0]), X]
print(X)

# set initial weights
W = [0 for i in range(X.shape[1])]
LR = 0.001

finished = False
count = 0

costArray = []

# iterations
while (not finished and count < 100000):
  # calculate gradient
  p = np.dot(X, W)
  error = p - Y
  gradient = np.dot(error, X)
  gradient /= len(Y)

  # update weights
  W = W - LR * gradient

  # calculate cost
  p = np.dot(X, W)
  cost = np.sum((p - Y) ** 2)
  cost /= len(Y)
  costArray.append(cost)

  # check if finished
  if (count > 0 and abs(costArray[count] - costArray[count - 1]) < 0.000001):
    finished = True
  else:
    count += 1

# plot costs over iterations
plt.plot(costArray)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()

print(f'Final Cost: {costArray[-1]}')
print(f'Final Weights: {W}')

# create a scatter plot of sugars vs. rating
plt.scatter(df['sugars'], df['rating'], s=8)
plt.xlabel('sugars')
plt.ylabel('rating')
plt.title('sugars vs. rating')
plt.show()

# create a scatter plot of rating vs. predicted rating
plt.scatter(df['rating'], np.dot(X, W), s=8)
plt.xlabel('rating')
plt.ylabel('predicted rating')
plt.title('rating vs. predicted rating')
plt.show()

# calculate R^2
p = np.dot(X, W)
error = p - Y
ssr = np.sum(error ** 2)
sst = np.sum((Y - Y.mean()) ** 2)
r2 = 1 - (ssr / sst)
print(f'R^2: {r2}')