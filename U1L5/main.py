import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the digits data set
digits = datasets.load_digits()
# Visualize an example digit image
plt.gray()
plt.matshow(digits.images[0]) # Use the first digit in the data set
plt.show()

# Extract the input data, force values to be between 0.0 and 1.0 (NORMALIZE the data)
X_digits = digits.data / digits.data.max()
# Extract the true values for each sample (each a digit between 0-9)
y_digits = digits.target
# Print the first 20 target values
y_digits[:20]

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=31)

#X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, train_size=0.9)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)

# Train the classifer
knn.fit(X_train, y_train)
# Compute the score (mean accuracy) on test set
score = knn.score(X_test, y_test)
print('KNN score: %f' % score)

pred = knn.predict(X_test)
print(pred)

correct = 0
for i in range(len(X_test)):
  if pred[i] != y_test[i]:
    correct += 1

print(f"Wrong Percent: {correct/len(X_test) * 100}%")