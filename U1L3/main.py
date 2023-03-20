import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('top2019.csv',engine='python')
print(df.head()) # See the top 5 rows of the dataframe

print(df['Genre'].describe())

new_df = df.sort_values(by="Length.")[:12]
print(new_df)

print(df.groupby('Loudness..dB..')['Energy'].count())

#plt.hist(df['Loudness..dB..'], bins=10)
#plt.show()

df = pd.read_csv('books.csv',engine='python')

# Make a scatter plot using matplotlib
fig, ax = plt.subplots()
ax.scatter(df['num_pages'],df['ratings_count']) # note this takes the form of (x,y)
# label axes and add a title
ax.set_title('Number of Pages vs. Ratings Count') # set the title
ax.set_xlabel('Number of Pages') # set a label for the x (horizontal) axis
ax.set_ylabel('Ratings Count') # set a label for the y (vertical) axis

fig, ax = plt.subplots()
ax.hist(df['ratings_count'], bins=10)
ax.set_title('Number of Ratings for each book') # set the title
ax.set_xlabel('Number of Ratings') # set a label for the x (horizontal) axis
ax.set_ylabel('Frequency (Books)') # set a label for the y (vertical) axis
plt.show()