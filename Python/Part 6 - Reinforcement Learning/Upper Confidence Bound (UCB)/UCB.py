# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

# Taking first 10 experiments for initialising
for i in range(0, 10):
    init = dataset.values[i, i]
    ads_selected.append(init)
    numbers_of_selections[i] = numbers_of_selections[i] + 1
    reward = dataset.values[i, i]
    sums_of_rewards[i] = sums_of_rewards[i] + reward
    total_reward = total_reward + reward
    
# Using other experiments for UCB
for n in range(10, N):
    ad = 0
    max_upper_bound = 0
    for j in range(0, d):
        average_reward = sums_of_rewards[j] / numbers_of_selections[j]
        delta_j = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[j])
        upper_bound = average_reward + delta_j
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = j
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()