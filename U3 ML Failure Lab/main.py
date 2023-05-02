# import statements
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker
from statsmodels.nonparametric.smoothers_lowess import lowess

#utility function
def convert_to_percentile(df, col_name):
    """Convert column to percentile.
    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    col_name : str
        Name of column in df to convert to percentile.

    Returns
    -------
    pd.Series
        Column converted to percentile from 1 to 100
    """
    return pd.qcut(df[col_name].rank(method='first'), 100,
                   labels=range(1, 101))

"""
Main Code
"""
data = pd.read_csv('health-care-bias-lab.csv')

"""
Medical Cost and Risk ---------------------
"""
# add a column of risk percentiles to the dataframe called 'risk_percentile'
risk_percentile = convert_to_percentile(data, "risk_score_t")
data["risk_percentile"] = risk_percentile
print("Dataframe with risk_percentile")
print(data.head())

# Create dataframe called `group_cost` with the mean total medical expenditure (`cost_t`) for each race (`race`) at each risk percentile (`risk_percentile`)
group_cost = data.groupby(["risk_percentile", "race"])[["cost_t"]].mean().reset_index()
print("Group_Cost dataframe")
print(group_cost.head())

# Divide `group_cost` into two dataframes based on race
#      Call the two dataframes `b_cost` and `w_cost`.
b_cost = group_cost[group_cost['race'] == 'black']
w_cost = group_cost[group_cost['race'] == 'white']

# Plot risk percentile against cost, splitting on race
ax = sns.scatterplot(x = "risk_percentile", y = "cost_t", data = group_cost, hue = "race", marker = "x", size = 2, legend = "full")
plt.yscale('log')
ax.set_yticks([1000, 3000, 8000, 20000, 60000])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend;
plt.show()

#fit a LOWESS (Locally Weighted Scatterplot Smoothing) model to the scatterplot above for each race
risk_percentile_array_b = np.array(b_cost['risk_percentile'])
risk_percentile_array_w = np.array(w_cost['risk_percentile'])
b_cost_array = np.array(b_cost['cost_t'])
w_cost_array = np.array(w_cost['cost_t'])
b_cost_lowess = lowess(b_cost_array, risk_percentile_array_b, it=35, frac=0.2, delta=2)
w_cost_lowess = lowess(w_cost_array, risk_percentile_array_w, it=35, frac=0.2, delta=2)

#plot the model on the scatterplot
ax = sns.scatterplot(x = "risk_percentile", y = "cost_t", data = group_cost, hue = "race", marker = "x", size = 2, legend = "full")
plt.yscale('log')
plt.plot(risk_percentile_array_b, b_cost_lowess[:, 1])
plt.plot(risk_percentile_array_w, w_cost_lowess[:, 1])
ax.set_yticks([1000, 3000, 8000, 20000, 60000])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()

""""
Chronic Illness and risk ---------------------------------------
"""
# Group the data by `risk_percentile` and `race`.
# Take the mean number of chronic illnesses (`gagne_sum_t`) in each group of race + risk percentile.
#      Call that dataframe `grouped_by_race`
# create dataframe with the average number of chronic illnesses for each race at each risk percentile
grouped_by_race = data.groupby(["risk_percentile", "race"])[["gagne_sum_t"]].mean().reset_index()
print("Grouped by race dataframe")
print(grouped_by_race.head())

# divide the grouped dataframe into two dataframes based on race
black_patients = grouped_by_race[grouped_by_race['race'] == 'black']
white_patients = grouped_by_race[grouped_by_race['race'] == 'white']

# Plot risk percentile against average number of chronic ilnesses, splitting on race
# create a scatterplot of risk percentile against average number of chronic ilnesses, splitting on race
sns.scatterplot(x="risk_percentile", y="gagne_sum_t", data=grouped_by_race, hue="race", marker="x", size=2, legend="full")
plt.show()

"""
Interaction between cost and illness
"""

# add a column of illness percentiles to the dataframe called 'illness_percentile'
illness_percentile = convert_to_percentile(data, "gagne_sum_t")
data['illness_percentile'] = illness_percentile
print('Dataframe after Adding illness_percentile')
print(data.head())

# Group the data by `illness_percentage` and `race`.
#      Take the mean total medical expenditure (`cost_t`) for each group of race + illness percentile.
#      Call the dataframe called `illnesses`
#create dataframe with the average total medical expenditure for each race at each illness percentile


# divide illnesses into two dataframes based on race: `illness_b` and `illness_w`
#divide illnesses into two dataframes based on race


# create a scatterplot of risk percentile against average number of chronic ilnesses, splitting on race
#scatterplot of illness percentile against cost, splitting on race


