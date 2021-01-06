import numpy as np
import pandas as pd

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


# In[2]:


##### Construct Utility Matrix ####### 
R = np.zeros((len(movies_description), len(users_description)))

for user, movie, rating in ratings_description.values:
    R[movie-1, user-1] = rating

R[R==0] = np.nan
print(f"Shape of Utility matrix is (movies, users): {R.shape}")


# In[ ]:


#### LATENT FACTORS HYPERPARAMETERS ####
from tqdm.auto import tqdm

EPOCHS = 1000
LEARNING_RATE = 0.05
K = 2 # number of factors to work with.

np.random.seed(42)
Q = np.random.uniform(-1, 1, (R.shape[0], K))
P = np.random.uniform(-1, 1, (K, R.shape[1]))
div = (R.shape[0] * R.shape[1]) - np.isnan(R).sum()
RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Starting RMSE: {RMSE}")

for epoch in tqdm(range(1000)):
    R_pred = np.matmul(Q,P)
    curr_error = np.nan_to_num(R - R_pred, 0)
    Q_update = np.zeros(Q.shape)
    for i in range(len(Q_update)):
        for curr_k in range(K):
            Q_delta =(-2 * np.dot(P[curr_k, :], curr_error[i]))/np.isnan(R[i]).sum()
            Q_update[i, curr_k] = LEARNING_RATE * Q_delta

    P_update = np.zeros(P.shape)
    for i in range(P_update.shape[1]):
        for curr_k in range(K):
            P_delta =(-2 * np.dot(Q[:, curr_k], curr_error[:, i]))/np.isnan(R[:, i]).sum()
            P_update[curr_k, i] = LEARNING_RATE * P_delta

    Q -= Q_update
    P -= P_update


RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0)**2).sum())/div)
print(f"Final RMSE: {RMSE}")


#### CREATE SUBMISSION ####
submission = []
for i, [user,movie] in enumerate(predictions_description.values):
    submission.append([i+1, R_pred[movie-1,user-1]])

submission_df = pd.DataFrame(submission, columns=["Id", "# Rating"])
submission_df.to_csv("data/submission.csv", index=False)

