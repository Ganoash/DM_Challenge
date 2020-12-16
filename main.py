import numpy as np
import pandas as pd
import os.path
from collaborative_filtering import run

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
# DATA IMPORT
##
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

utility_matrix: pd.DataFrame = \
    ratings_description.pivot(index='userID', columns='movieID', values='rating')


#####
##
# COLLABORATIVE FILTERING
##
#####
def predict_collaborative_filtering(utility):
    to_parse = run(utility).values.tolist()
    predictions = {}
    for index, item in enumerate(to_parse):
        predictions[index] = item[0]
    predictions = pd.DataFrame(predictions.values(), index=predictions.keys())
    ret_value = pd.DataFrame(predictions_description["movieID"]).applymap(lambda x: predictions.loc[x].item())
    ret_value["index"] = ret_value.index
    ret_index = ret_value["index"]+1
    ret_value = ret_value["movieID"]

    return zip(ret_index.values.astype(int).tolist(), ret_value.values.tolist())


#####
##
# LATENT FACTORS
##
#####

def predict_latent_factors(utility):
    learning_rate = 0.05
    k = 2

    np.random.seed(42)
    Q = np.random.uniform(-1, 1, (utility.shape[0], k))
    P = np.random.uniform(-1, 1, (k, utility.shape[1]))
    div = (utility.shape[0] * utility.shape[1]) - np.isnan(utility).sum()
    RMSE = np.sqrt(((np.nan_to_num(utility - np.matmul(Q, P), 0) ** 2).sum()) / div)
    print(f"Starting RMSE: {RMSE}")

    for epoch in range(1000):
        prediction = np.matmul(Q, P)
        curr_error = np.nan_to_num(utility_matrix - prediction, 0)
        Q_update = np.zeros(Q.shape)
        for i in range(len(Q_update)):
            for curr_k in range(k):
                Q_delta = (-2 * np.dot(P[curr_k, :], curr_error[i])) / np.isnan(utility.iloc[i]).sum()
                Q_update[i, curr_k] = learning_rate * Q_delta

        P_update = np.zeros(P.shape)
        for i in range(P_update.shape[1]):
            for curr_k in range(k):
                P_delta = (-2 * np.dot(Q[:, curr_k], curr_error[:, i])) / np.isnan(utility.iloc[:, i]).sum()
                P_update[curr_k, i] = learning_rate * P_delta

        Q -= Q_update
        P -= P_update
    return prediction


#####
##
# FINAL PREDICTORS
##
#####

def predict_final(prediction, utility):
    return predict_collaborative_filtering(utility)


#####
##
# RANDOM PREDICTORS
# //!!\\ TO CHANGE
##
#####


#####
##
# SAVE RESULTS
##
#####

# //!!\\ TO CHANGE by your prediction function
predictions = predict_final(predictions_description, utility_matrix)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it down
    submission_writer.write(predictions)
