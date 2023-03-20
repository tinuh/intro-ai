import pandas as pd
import numpy as np
from scipy import stats

# read the books data
df = pd.read_csv('books.csv', engine='python')

# Question 1
# remove all outliers 3 standard deviations away from the mean using the z-score
print(f'Number of average_ratings: {df["average_rating"].count()}')
df = df[(np.abs(stats.zscore(df['average_rating'])) < 3)]
print(f'Number of average_ratings (after removing outliers): {df["average_rating"].count()}')

# Question 2
# create a new column called RatingsToReviews
df['RatingsToReviews'] = df['ratings_count'] / df['text_reviews_count']
# when dividing by 0, values result in infinity or -infinity, replace these values with NaN
df = df.replace([np.inf, -np.inf], np.nan)
print(df[['ratings_count', 'text_reviews_count', 'RatingsToReviews']].head(10))

# Question 3
# create a normalized and standerdized version of the average_rating column
df['average_rating_normalized'] = (df['average_rating'] - df['average_rating'].min()) / (df['average_rating'].max() - df['average_rating'].min())
df['average_rating_standardized'] = (df['average_rating'] - df['average_rating'].mean()) / df['average_rating'].std()
print(df[['average_rating', 'average_rating_normalized', 'average_rating_standardized']].head(10))