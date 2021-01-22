#!/usr/bin/env python
# coding: utf-8


from collaborative_filtering import run
import numpy as np
import pandas as pd
import os.path
from numpy import genfromtxt

# def predict_collaborative_filtering(utility):
def predict_final(utility):
    return run(utility, predictions_description)


# Where data is located
movies_file = "./data/movies.csv"
users_file = "./data/users.csv"
ratings_file = "./data/ratings.csv"
predictions_file = "./data/predictions.csv"
submission_file = "./data/submission.csv"

# Read the data using pandas
movies_description = pd.read_csv(
    movies_file,
    delimiter=";",
    dtype={"movieID": "int", "year": "int", "movie": "str"},
    names=["movieID", "year", "movie"],
)
users_description = pd.read_csv(
    users_file,
    delimiter=";",
    dtype={"userID": "int", "gender": "str", "age": "int", "profession": "int"},
    names=["userID", "gender", "age", "profession"],
)
ratings_description = pd.read_csv(
    ratings_file,
    delimiter=";",
    dtype={"userID": "int", "movieID": "int", "rating": "int"},
    names=["userID", "movieID", "rating"],
)
predictions_description = pd.read_csv(
    predictions_file, delimiter=";", names=["userID", "movieID"], header=None
)

utility_matrix: pd.DataFrame = ratings_description.pivot(
    index="userID", columns="movieID", values="rating"
)


utility_matrix.loc[
    :,
    set(movies_description["movieID"].to_numpy().tolist()).difference(
        set(utility_matrix.columns.to_numpy().tolist())
    ),
] = 0
print(utility_matrix.shape)


train_predictions = ratings_description[["userID", "movieID"]]


# Loaded from Latent_Factors_Basic.ipynb
Q_lf = genfromtxt("./latent_factors/Q.csv")
P_lf = genfromtxt("./latent_factors/P.csv")
R_pred = np.matmul(Q_lf, P_lf)
cf_train_predictions = dict(run(utility_matrix, train_predictions))


#### CREATE SUBMISSION ####
lf_train_predictions = []
for i, [user, movie, rating] in enumerate(ratings_description.values):
    lf_train_predictions.append([i + 1, R_pred[movie - 1, user - 1]])


labels = ratings_description.values[:, -1]
lf_train = np.array(lf_train_predictions)

cf_train = np.array([[k, v] for k, v in cf_train_predictions.items()])


def rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).sum() / len(actual))


pred_curve = []
for a in np.arange(0.0, 1.0, 0.01):
    tmp_predictions = a * lf_train[:, -1] + (1 - a) * cf_train[:, -1]
    pred_curve.append([a, rmse(labels, tmp_predictions)])
pred_curve = np.array(pred_curve)


import matplotlib.pyplot as plt

plt.figure(figsize=(30, 10))
plt.title("Ensemble Model prediction curve", fontsize=20)
plt.xlabel("a", fontsize=20)
plt.ylabel("RMSE", fontsize=25)
plt.plot(pred_curve[:, 0], pred_curve[:, 1])
