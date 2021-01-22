#!/usr/bin/env python
# coding: utf-8

# ### Construct Utility Matrix 

# In[8]:


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


# ## "Normal" Latent Factor Model

# In[14]:


from numpy import genfromtxt

# Loaded from Latent_Factors_Basic.ipynb
Q_lf = genfromtxt("./latent_factors/Q.csv")
P_lf = genfromtxt("./latent_factors/P.csv")
objective_lf = genfromtxt("./latent_factors/objectives.csv")


# ## Latent Factors with Regularization

# In[ ]:


from numpy import genfromtxt

# Loaded from Latent_Factors_Regularized.ipynb
Q_reg = genfromtxt("./lf_reg/Q.csv")
P_reg= genfromtxt("./lf_reg/P.csv")
objective_reg = genfromtxt("./lf_reg/objectives.csv")


# ## Latent Factors with Stochastic Gradient Descent (regularized)

# In[15]:


from numpy import genfromtxt

# Loaded from Latent_Factors_SGD_Regularized.ipynb
Q_sgd = genfromtxt("./lf_sgd/Q.csv")
P_sgd= genfromtxt("./lf_sgd/P.csv")
objective_sgd = genfromtxt("./lf_sgd/objectives.csv")


# In[85]:


from numpy import genfromtxt

# Loaded from Latent_Factors_SGD_Regularized.ipynb
objective_bias = genfromtxt("./lf_bias/objectives.csv")


# ## Latent Factor (SGD + biases + regularization)

EPOCHS = 300
LEARNING_RATE = 0.00002
LEARNING_RATE_bias = 0.00002
LAMBDA = 0.2
LAMBDA_bias = 0.2


K = 2 # number of factors to work with.

np.random.seed(42)

ratings = ratings_description.values.copy()
ratings[:, 0:2] = ratings[:, 0:2] - 1

# user movie interaction
Q = np.random.uniform(-2, 2, (R.shape[0], K)) # movies
P = np.random.uniform(-2, 2, (K, R.shape[1])) # users
bias_learning_curve = []

# biases
tmp_R = np.nan_to_num(R)
overall_bias = np.mean(tmp_R[tmp_R!=0])

user_bias = [np.mean(user_ratings[user_ratings!=0]) for user_ratings in tmp_R.T] - overall_bias

movie_bias = [np.mean(movie_ratings[movie_ratings!=0]) for movie_ratings in tmp_R] - overall_bias 

overall_bias = np.ones((num_movies, num_users)) * overall_bias

div = (R.shape[0] * R.shape[1]) - np.isnan(R).sum()
R_pred_pre = overall_bias + np.tile(user_bias, (num_movies, 1)) +  np.tile(movie_bias, (num_users, 1)).T + np.matmul(Q, P)
RMSE = np.sqrt(((np.nan_to_num(R - R_pred_pre, 0)**2).sum())/div)
print(f"Starting RMSE: {RMSE}")

for epoch in range(EPOCHS):
    np.random.shuffle(ratings) # inplace shuffle of matrix
    R_pred = overall_bias + np.tile(user_bias, (num_movies, 1)) +  np.tile(movie_bias, (num_users, 1)).T + np.matmul(Q, P)
    curr_error = 2*np.nan_to_num(R - R_pred, 0)
    for userID, movieID, rating in ratings:
        q_update = LEARNING_RATE * (curr_error[movieID, userID]*P[:, userID] - LAMBDA*Q[movieID, :])
        Q[movieID, :] = Q[movieID, :] + q_update

        p_update = LEARNING_RATE * (curr_error[movieID, userID]*Q[movieID, :] - LAMBDA*P[:, userID])
        P[:, userID] = P[:, userID] + p_update
        
        movie_bias_update = LEARNING_RATE * (curr_error[movieID,userID] - LAMBDA_bias*movie_bias[movieID])
        movie_bias[movieID] = movie_bias[movieID] + movie_bias_update
        
        user_bias_update = LEARNING_RATE * (curr_error[movieID, userID] - LAMBDA_bias*user_bias[userID])
        user_bias[userID] = user_bias[userID] + user_bias_update
    
    R_pred_post = overall_bias + np.tile(user_bias, (num_movies, 1)) +  np.tile(movie_bias, (num_users, 1)).T + np.matmul(Q, P)
    RMSE_i = np.sqrt(((np.nan_to_num(R - R_pred_post, 0)**2).sum())/div)
    print(f"RMSE {epoch}: {RMSE_i}")
    bias_learning_curve.append([epoch, RMSE_i])


R_pred_post = overall_bias + np.tile(user_bias, (num_movies, 1)) +  np.tile(movie_bias, (num_users, 1)).T + np.matmul(Q, P)
RMSE = np.sqrt(((np.nan_to_num(R - R_pred_post, 0)**2).sum())/div)
print(f"Final RMSE: {RMSE}")


# In[43]:


from numpy import savetxt
from pathlib import Path

Path("./lf_bias2/").mkdir(parents=True, exist_ok=True)


savetxt("lf_bias2/Q.csv", Q)
savetxt("lf_bias2/P.csv", P)
savetxt("lf_bias2/moviebias.csv", movie_bias)
savetxt("lf_bias2/userbias.csv", user_bias)

savetxt("lf_bias2/objectives.csv", bias_learning_curve)


# In[89]:


import matplotlib.pyplot as plt
plt.figure(figsize=(30, 10))
plt.title("Latent Factor Model - Learning Curve Comparison", fontsize=20)
plt.xlabel("epoch", fontsize=20)
plt.ylabel("RMSE", fontsize=25)
plt.plot(objective_lf[:, 0], objective_lf[:, 1], label="'Normal' Latent Factor Model")
# plt.plot(objective_reg[:, 0], objective_reg[:, 1], label="Latent Factor w. Regularization")
plt.plot(objective_sgd[:190, 0], objective_sgd[:190, 1], label="Latent Factor SGD + Regularization")
plt.plot(objective_bias[:, 0], objective_bias[:, 1], label="Latent Factor SGD + Regularization + bias")
fig = plt.legend(loc="upper right", prop={'size': 20})


#### CREATE SUBMISSION ####
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
submission = []
R_pred = overall_bias + np.tile(user_bias, (num_movies, 1)) +  np.tile(movie_bias, (num_users, 1)).T +  np.matmul(Q, P)
for i, [user,movie] in enumerate(predictions_description.values):
    submission.append([i+1, R_pred[movie-1,user-1]])

submission_df = pd.DataFrame(submission, columns=["Id", "Rating"])
submission_df = submission_df.fillna(overall_bias[0, 0])
submission_df.to_csv("data/submission.csv", index=False)

