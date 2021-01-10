from typing import Dict

import pandas as pd
import numpy as np

feature_dict: Dict[str, pd.Series] = {}
feature_user_dict: Dict[str, pd.Series] = {}


def feature_bias(movies: pd.DataFrame, utility: pd.DataFrame, feature: str):
    # take collumn mean and allign movies index to utility index
    feature_means = pd.DataFrame(utility.mean(axis=0))
    feature_means = feature_means.reindex(sorted(feature_means.index), axis=0)
    # match collumn mean with feature
    feature_means[feature] = movies[feature]

    # group by feature and take the mean value per feature
    feature_dict[feature] = feature_means.groupby(feature).mean()

    return feature_dict[feature]


"""
method for calculating how an item relates to a certain feature. takes the movie matrix with the feature collumn
a utility matrix with all user_item matrices, the name of the given feature and the item index
:return a 
"""


def item_feature_bias(movies: pd.DataFrame, utility: pd.DataFrame, feature: str, movieID: int):
    # fix off by one error
    # calculate feature matrix if it doesn't exist yet
    if not (feature in feature_dict):
        feature_bias(movies, utility, feature)

    movie = movies.loc[movieID - 1]

    # take feature array for the given feature, select the index where the feature for the given movie matches,
    # and retrieve the value
    return utility.mean(axis=1).loc[movieID] - \
           feature_dict[feature].loc[feature_dict[feature].index == movie[feature]].values[0][0]


"""
method for calculating how a user relates to a certain feature. takes the movie matrix with the feature collumn
a utility matrix with all user_item matrices, the name of the given feature
:return 
"""


def user_feature_bias(movies: pd.DataFrame, utility: pd.DataFrame, feature: str, userID: int, movieID: int):
    user = utility.loc[userID]
    if not (feature in feature_dict):
        feature_bias(movies, utility, feature)

    if not (feature in feature_user_dict):
        feature_user_dict[feature] = {}

    if not (user.name in feature_user_dict[feature]):
        feature_means = pd.DataFrame(user)
        feature_means[feature] = movies[feature]
        feature_user_dict[feature][user.name] = feature_means.groupby(feature).mean()

    movie = movies.loc[movieID - 1]
    # first get the feature dict for this specific user, and select the specified feature rating
    # then subtract the average feature rating.
    print("user feature rating: " + str(
        feature_user_dict[feature][user.name].loc[feature_user_dict[feature][user.name].index == movie[feature]].values[
            0][0]))
    print("average feature rating " + str(
        feature_dict[feature].loc[feature_dict[feature].index == movie[feature]].values[0][0]))
    return feature_user_dict[feature][user.name].loc[
               feature_user_dict[feature][user.name].index == movie[feature]].values[0][0] - \
           feature_dict[feature].loc[feature_dict[feature].index == movie[feature]].values[0][0]

    # movies['mean_year'] = means_per_year.groupby('year').mean()
