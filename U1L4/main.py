from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix

# data procesing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# set pandas to display all columns
pd.set_option('display.max_columns', None)

# load the iris dataset and split into data and labels
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=31)

#create a pandas dataframe
df = pd.DataFrame(data=np.c_[X, y])

print(df.head(10))

#create a scatter plot
fig, ax = plt.subplots()
ax.scatter(df[2], df[3])
# label axes and add a title
ax.set_title('Petal Length vs. Petal Width') # set the title
ax.set_xlabel('Petal Length (cm)') # set a label for the x (horizontal) axis
ax.set_ylabel('Petal Width (cm)') # set a label for the y (vertical) axis

plt.show()

# create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, y_train)

fn= ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
# This label will be used later in the confusion matrix
predn = ['setosa (pred)', 'versicolor (pred)', 'virginica (pred)']

tree.plot_tree(clf)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi=300)
tree.plot_tree(clf, feature_names = fn, class_names=cn, filled = True)

fig.savefig('tree.png')
plt.show()

y_predict = clf.predict(X_test)
print(y_predict)

acc = clf.score(X_test, y_test)
print(f"Accuracy: {acc}")

fig, ax = plt.subplots()
ax.scatter(X_test[:,0], X_test[:,1], c=y_predict) # note this takes the form of (x,y)
# label axes and add a title
ax.set_title('Sepal Length vs. Sepal Width') # set the title
ax.set_xlabel('Sepal Length (cm)') # set a label for the x (horizontal) axis
ax.set_ylabel('Sepal Width (cm)') # set a label for the y (vertical) axis

plt.show()

#create a confusion matrix using the test data
y_predict = clf.predict(X_train)
confusion_matrix = confusion_matrix(y_train, y_predict)
cm_df = pd.DataFrame(confusion_matrix, index = cn, columns = predn)
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.show()