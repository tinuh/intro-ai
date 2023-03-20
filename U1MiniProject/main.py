import pandas as pd
from matplotlib import pyplot as plt
from sklearn import neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# read the csv file into a dataframe
df = pd.read_csv('Airline.csv', engine='python')

# See the top 5 rows of the dataframe
# print(df.head())

# convert features to numeric values
df['Customer Type'].replace(['Loyal Customer', 'disloyal Customer'], [0, 1], inplace=True)
df['Class'].replace(['Eco', 'Eco Plus', 'Business'], [0, 1, 2], inplace=True)
df['satisfaction'].replace(['neutral or dissatisfied', 'satisfied'], [0, 1], inplace=True)

# split the data into a test set and training set with X being the features and y being the labels
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Class', 'Customer Type', 'Seat comfort', 'Leg room service']], df['satisfaction'], test_size=0.2)

# use the Decision Tree Classifier to train the model
clf = DecisionTreeClassifier(max_depth=8)
clf = clf.fit(X_train, y_train)

# knn = neighbors.KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train, y_train)

#plot the decision tree
# fn=['Age', 'Class', 'Customer Type', 'Seat comfort', 'Leg room service']
# cn=['neutral or dissatisfied', 'satisfied']

# tree.plot_tree(clf)
# fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi=300)
# tree.plot_tree(clf, feature_names = fn, class_names=cn, filled = True)

# plt.show()

# create a segmented histogram of age colored by satisfaction (grouped into 2 bins so hard to read)
# plt.hist([df[df['satisfaction'] == 0]['Age'], df[df['satisfaction'] == 1]['Age']], stacked=True, color=['r', 'b'], bins=30, label=['neutral or dissatisfied', 'satisfied'])
# plt.xlabel('Age')
# plt.ylabel('Number of passengers')
# plt.legend()
# plt.show()

# create a stacked bar chart of the class segmented by satisfaction
df['Class'].replace([0, 1, 2] , ['Eco', 'Eco Plus', 'Business'], inplace=True) # convert values back to strings for accuralate labels on bar chart
df.groupby(['Class', 'satisfaction']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Class vs. Satisfaction')
plt.xlabel('Class')
plt.ylabel('Number of passengers')
plt.legend(['neutral or dissatisfied', 'satisfied'])
plt.show()

# get the accuracy of the model by testing it using the test set
score = clf.score(X_test, y_test)
print(f'Decision Tree Accuracy: {score * 100}%')