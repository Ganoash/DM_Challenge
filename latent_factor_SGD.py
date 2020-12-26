#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
# DATA IMPORT
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

utility_matrix: pd.DataFrame =         ratings_description.pivot(index='userID', columns='movieID', values='rating').T
utility_matrix


# In[9]:


##### Construct Utility Matrix ####### 
R = np.zeros((len(movies_description), len(users_description)))

for user, movie, rating in ratings_description.values:
    R[movie-1, user-1] = rating

R[R==0] = np.nan
print(f"Shape of Utility matrix is (movies, users): {R.shape}")


# In[ ]:


from tqdm.auto import tqdm

EPOCHS = 100
LEARNING_RATE = 0.0001
LAMBDA = 0.1


K = 2 # number of factors to work with.

# np.random.seed(42)

# ratings = ratings_description.values.copy()
# ratings[:, 0:2] = ratings[:, 0:2] - 1
# Q = np.random.uniform(-1, 1, (R.shape[0], K)) # movies
# P = np.random.uniform(-1, 1, (K, R.shape[1])) # users
# div = (R.shape[0] * R.shape[1]) - np.isnan(R).sum()
# RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
# print(f"Starting RMSE: {RMSE}")

# #### LATENT FACTORS HYPERPARAMETERS ####

# objectives = []
for epoch in tqdm(range(EPOCHS)):
#     np.random.shuffle(ratings) # inplace shuffle of matrix
    R_pred = np.matmul(Q, P)
    print(np.min(Q), np.max(Q))
    curr_error = 2*np.nan_to_num(R - R_pred, 0)
    for userID, movieID, rating in ratings:
        q_update = LEARNING_RATE * (curr_error[movieID, userID]*P[:, userID] - LAMBDA*Q[movieID, :])
        Q[movieID, :] = Q[movieID, :] + q_update

        p_update = LEARNING_RATE * (curr_error[movieID, userID]*Q[movieID, :] - LAMBDA*P[:, userID])
        P[:, userID] = P[:, userID] + p_update

    RMSE_i = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
    print(f"RMSE {epoch}: {RMSE_i}")
    objectives.append([epoch, RMSE_i])

RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Final RMSE: {RMSE}")


# In[40]:


import matplotlib.pyplot as plt
plt.figure(figsize=(30, 10))
plt.title("Learning curve - SGD Latent Factor Model", fontsize=20)
plt.xlabel("epoch", fontsize=20)
plt.ylabel("RMSE", fontsize=25)
plt.plot(np.array(objectives)[-20:, 1])
plt.savefig("figures/SGD-latent-factor.png")


# In[43]:


#### CREATE SUBMISSION ####
submission = []
R_pred = np.matmul(Q, P)
for i, [user,movie] in enumerate(predictions_description.values):
    submission.append([i+1, R_pred[movie-1,user-1]])

submission_df = pd.DataFrame(submission, columns=["Id", "Rating"])
submission_df.to_csv("data/submission.csv", index=False)

