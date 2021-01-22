#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os.path
from random import randint

ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'

ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])

num_movies = max(ratings_description["movieID"])
num_users = max(ratings_description["userID"])
R = np.zeros(( num_movies, num_users))

for user, movie, rating in ratings_description.values:
    R[movie-1, user-1] = rating

R[R==0] = np.nan
print(f"Shape of Utility matrix is (movies, users): {R.shape}")
R


# In[9]:


from tqdm.auto import tqdm

EPOCHS = 200
LEARNING_RATE = 0.00002
LAMBDA = 0.2

K = 2 # number of factors to work with.

np.random.seed(42)

ratings = ratings_description.values.copy()
ratings[:, 0:2] = ratings[:, 0:2] - 1
# Q = np.random.uniform(-1, 1, (R.shape[0], K)) # movies
# P = np.random.uniform(-1, 1, (K, R.shape[1])) # users
# sgd_learning_curve = []

div = (R.shape[0] * R.shape[1]) - np.isnan(R).sum()
RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Starting RMSE: {RMSE}")

for epoch in tqdm(range(EPOCHS)):
    np.random.shuffle(ratings) # inplace shuffle of matrix
    R_pred = np.matmul(Q, P)
    curr_error = 2*np.nan_to_num(R - R_pred, 0)
    for userID, movieID, rating in ratings:
        q_update = LEARNING_RATE * (curr_error[movieID, userID]*P[:, userID] - LAMBDA*Q[movieID, :])
        Q[movieID, :] = Q[movieID, :] + q_update

        p_update = LEARNING_RATE * (curr_error[movieID, userID]*Q[movieID, :] - LAMBDA*P[:, userID])
        P[:, userID] = P[:, userID] + p_update

    RMSE_i = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
    print(f"RMSE {epoch}: {RMSE_i}")
    sgd_learning_curve.append([epoch, RMSE_i])

RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Final RMSE: {RMSE}")


# In[7]:


from numpy import savetxt
from pathlib import Path

Path("./lf_sgd/").mkdir(parents=True, exist_ok=True)


savetxt("lf_sgd/Q.csv", Q)
savetxt("lf_sgd/P.csv", P)
savetxt("lf_sgd/objectives.csv", sgd_learning_curve)


# In[8]:


#### CREATE SUBMISSION ####
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
submission = []
R_pred = np.matmul(Q, P)
for i, [user,movie] in enumerate(predictions_description.values):
    submission.append([i+1, R_pred[movie-1,user-1]])

submission_df = pd.DataFrame(submission, columns=["Id", "Rating"])
submission_df.to_csv("lf_sgd/submission.csv", index=False)

