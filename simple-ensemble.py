#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collaborative_filtering import run

def predict_collaborative_filtering(utility):
    return run(utility, predictions_description)


# In[4]:


import numpy as np
import pandas as pd
import os.path
from collaborative_filtering import run

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

utility_matrix: pd.DataFrame =     ratings_description.pivot(index='userID', columns='movieID', values='rating')


utility_matrix.loc[:, set(movies_description['movieID'].to_numpy().tolist()).difference(set(utility_matrix.columns.to_numpy().tolist()))] = 0
print(utility_matrix.shape)


# In[12]:


train_predictions = ratings_description[["userID", "movieID"]]


# In[16]:


get_ipython().run_cell_magic('time', '', 'cf_train_predictions = dict(run(utility_matrix, train_predictions))')


# In[22]:


##### Construct Utility Matrix ####### 
R = np.zeros((len(movies_description), len(users_description)))

for user, movie, rating in ratings_description.values:
    R[movie-1, user-1] = rating

R[R==0] = np.nan
print(f"Shape of Utility matrix is (movies, users): {R.shape}")

#### LATENT FACTORS WITH REGULARIZATION ####
#### LATENT FACTORS HYPERPARAMETERS ####
from tqdm import tqdm_notebook as tqdm

EPOCHS = 1000
LEARNING_RATE = 0.05 # == nu
LAMBDA = 0.01
K = 2 # number of factors to work with.

np.random.seed(42)
Q = np.random.uniform(-1, 1, (R.shape[0], K))
P = np.random.uniform(-1, 1, (K, R.shape[1]))
div = (R.shape[0] * R.shape[1]) - np.isnan(R).sum()
RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Starting RMSE: {RMSE}")

objectives = []

for epoch in tqdm(range(1000)):
    R_pred = np.matmul(Q,P)
    curr_error = np.nan_to_num(R - R_pred, 0)
    Q_update = np.zeros(Q.shape)
    for i in range(len(Q_update)):
        for curr_k in range(K):
            Q_delta =(-2 * np.dot(P[curr_k, :], curr_error[i]))/np.isnan(R[i]).sum()
            Q_update[i, curr_k] = LEARNING_RATE * (Q_delta + LAMBDA*Q[i, curr_k])

    P_update = np.zeros(P.shape)
    for i in range(P_update.shape[1]):
        for curr_k in range(K):
            P_delta =(-2 * np.dot(Q[:, curr_k], curr_error[:, i]))/np.isnan(R[:, i]).sum()
            P_update[curr_k, i] = LEARNING_RATE * (P_delta + LAMBDA*P[curr_k, i])

    Q -= Q_update
    P -= P_update
    
    RMSE_i = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
    print(f"RMSE {epoch}: {RMSE_i}")
    objectives.append([epoch, RMSE_i])


RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Final RMSE: {RMSE}")


# In[26]:


#### CREATE SUBMISSION ####
lf_train_predictions = []
for i, [user,movie, rating] in enumerate(ratings_description.values):
    lf_train_predictions.append([i+1, R_pred[movie-1,user-1]])


# In[39]:


labels = ratings_description.values[:, -1]
lf_train = np.array(lf_train_predictions)

cf_train = np.array([[k,v] for k, v in cf_train_predictions.items()])


# In[40]:


labels


# In[41]:


lf_train


# In[42]:


cf_train


# In[43]:


def rmse(actual, predicted):
    return np.sqrt(((actual-predicted)**2).sum()/len(actual))


# In[50]:


pred_curve = []
for a in np.arange(0.0, 1.0, 0.01):
    tmp_predictions = a*lf_train[:, -1] + (1-a)*cf_train[:, -1]
    pred_curve.append([a, rmse(labels, tmp_predictions)])
pred_curve = np.array(pred_curve)


# In[51]:


pred_curve


# In[52]:


import matplotlib.pyplot as plt
plt.figure(figsize=(30, 10))
plt.title("Ensemble Model prediction curve", fontsize=20)
plt.xlabel("a", fontsize=20)
plt.ylabel("RMSE", fontsize=25)
plt.plot(pred_curve[:, 0], pred_curve[:, 1])      

