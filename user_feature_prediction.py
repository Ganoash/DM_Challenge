from typing import Dict

import pandas as pd
import numpy as np

feature_dict: Dict[str, pd.Series] = {}
feature_movie_dict: Dict[str, pd.Series] = {}


def feature_bias(user: pd.DataFrame, utility: pd.DataFrame, feature: str):
    # take collumn mean and allign movies index to utility index
    feature_means = pd.DataFrame(utility.mean(axis=1))
    feature_means = feature_means.reindex(sorted(feature_means.index), axis=1)
    # match collumn mean with feature
    feature_means[feature] = user[feature]

    # group by feature and take the mean value per feature
    feature_dict[feature] = feature_means.groupby(feature).mean()

    return feature_dict[feature]


"""
method for calculating how an item relates to a certain feature. takes the movie matrix with the feature collumn
a utility matrix with all user_item matrices, the name of the given feature and the item index
:return a 
"""


def item_feature_bias(user: pd.DataFrame, utility: pd.DataFrame, feature: str, userID: int, movieID: int):
    movie = utility.loc[movieID]

    if not (feature in feature_dict):
        feature_bias(user, utility, feature)

    if not (feature in feature_movie_dict):
        feature_movie_dict[feature] = {}

    if not (movie.name in feature_movie_dict[feature]):
        feature_means = pd.DataFrame(movie)
        feature_means[feature] = user[feature]
        feature_movie_dict[feature][user.name] = feature_means.groupby(feature).mean()

    users = user.loc[userID - 1]
    # first get the feature dict for this specific movie, and select the specified feature rating
    # then subtract the average feature rating.
    print("user feature rating: " + str(
        feature_movie_dict[feature][movie.name].loc[
            feature_movie_dict[feature][movie.name].index == users[feature]].values[
            0][0]))
    print("average feature rating " + str(
        feature_dict[feature].loc[feature_dict[feature].index == user[feature]].values[0][0]))
    return \
        feature_movie_dict[feature][movie.name].loc[
            feature_movie_dict[feature][movie.name].index == users[feature]].values[
            0][0] \
        - feature_dict[feature].loc[feature_dict[feature].index == user[feature]].values[0][0]


"""
method for calculating how a user relates to a certain feature. takes the movie matrix with the feature collumn
a utility matrix with all user_item matrices, the name of the given feature
:return 
"""


def user_feature_bias(users: pd.DataFrame, utility: pd.DataFrame, feature: str, userID: int):
    # calculate feature matrix if it doesn't exist yet
    if not (feature in feature_dict):
        feature_bias(users, utility, feature)

    # fix off by one error
    user = users.loc[userID - 1]

    # take feature array for the given feature, select the index where the feature for the given movie matches,
    # and retrieve the value
    return utility.mean(axis=1).loc[users] - feature_dict[feature].loc[
        feature_dict[feature].index == user[feature]].values[0][0]

    # movies['mean_year'] = means_per_year.groupby('year').mean()
