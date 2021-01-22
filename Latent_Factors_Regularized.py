#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os.path
from random import randint
from numpy import savetxt
from pathlib import Path

ratings_file = "./data/ratings.csv"
predictions_file = "./data/predictions.csv"

ratings_description = pd.read_csv(
    ratings_file,
    delimiter=";",
    dtype={"userID": "int", "movieID": "int", "rating": "int"},
    names=["userID", "movieID", "rating"],
)

num_movies = max(ratings_description["movieID"])
num_users = max(ratings_description["userID"])
R = np.zeros((num_movies, num_users))

for user, movie, rating in ratings_description.values:
    R[movie - 1, user - 1] = rating

R[R == 0] = np.nan
print(f"Shape of Utility matrix is (movies, users): {R.shape}")


EPOCHS = 1500
LEARNING_RATE = 0.00002  # == nu
LAMBDA = 0.2
K = 2  # number of factors to work with.

Q = np.random.uniform(-1, 1, (R.shape[0], K))
P = np.random.uniform(-1, 1, (K, R.shape[1]))
regularization_learning_curve = []

div = (R.shape[0] * R.shape[1]) - np.isnan(R).sum()
RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0) ** 2).sum()) / div)
print(f"Starting RMSE: {RMSE}")


for epoch in range(EPOCHS):
    R_pred = np.matmul(Q, P)
    curr_error = np.nan_to_num(R - R_pred, 0)
    Q_update = np.zeros(Q.shape)
    for i in range(len(Q_update)):
        for curr_k in range(K):
            Q_delta = (-2 * np.dot(P[curr_k, :], curr_error[i])) / np.isnan(R[i]).sum()
            Q_update[i, curr_k] = LEARNING_RATE * (Q_delta + LAMBDA * Q[i, curr_k])

    P_update = np.zeros(P.shape)
    for i in range(P_update.shape[1]):
        for curr_k in range(K):
            P_delta = (-2 * np.dot(Q[:, curr_k], curr_error[:, i])) / np.isnan(
                R[:, i]
            ).sum()
            P_update[curr_k, i] = LEARNING_RATE * (P_delta + LAMBDA * P[curr_k, i])

    Q -= Q_update
    P -= P_update

    RMSE_i = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0) ** 2).sum()) / div)
    print(f"RMSE {epoch}: {RMSE_i}")
    regularization_learning_curve.append([epoch, RMSE_i])


RMSE = np.sqrt(((np.nan_to_num(R - np.matmul(Q, P), 0) ** 2).sum()) / div)
print(f"Final RMSE: {RMSE}")


Path("./lf_reg/").mkdir(parents=True, exist_ok=True)


savetxt("lf_reg/Q.csv", Q)
savetxt("lf_reg/P.csv", P)
savetxt("lf_reg/objectives.csv", regularization_learning_curve)


#### CREATE SUBMISSION ####
predictions_description = pd.read_csv(
    predictions_file, delimiter=";", names=["userID", "movieID"], header=None
)
submission = []
R_pred = np.matmul(Q, P)
for i, [user, movie] in enumerate(predictions_description.values):
    submission.append([i + 1, R_pred[movie - 1, user - 1]])

submission_df = pd.DataFrame(submission, columns=["Id", "Rating"])
submission_df.to_csv("lf_reg/submission.csv", index=False)
