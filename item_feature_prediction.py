import pandas as pd
import numpy as np

feature_dict = {}
feature_user_dict = {}


def feature_bias(movies: pd.DataFrame, utility: pd.DataFrame, feature: str, ):
    # take collumn mean and allign movies index to utility index
    feature_means = pd.DataFrame(utility.mean(axis=0))
    feature_means = feature_means.reindex(sorted(feature_means.index), axis=0)
    # match collumn mean with feature
    feature_means[feature] = movies[feature]

    # group by feature and take the mean value per feature
    feature_dict[feature] = feature_means.groupby(feature).mean()


"""
method for calculating how an item relates to a certain feature. takes the movie matrix with the feature collumn
a utility matrix with all user_item matrices, the name of the given feature and the item index
:return a 
"""


def item_feature_bias(movies: pd.DataFrame, utility: pd.DataFrame, feature: str, index: int):
    # calculate feature matrix if it doesn't exist yet
    if feature_dict[feature] is None:
        feature_bias(movies, utility, feature)

    return feature_dict[feature] - utility.mean(axis=0).loc[index]


"""
method for calculating how a user relates to a certain feature. takes the movie matrix with the feature collumn
a utility matrix with all user_item matrices, the name of the given feature
:return 
"""


def user_year_bias(movies: pd.DataFrame, utility: pd.DataFrame, user: pd.Series, feature: str):
    if feature_dict[feature] is None:
        feature_bias(movies, utility, feature)
    f_bias = feature_dict[feature]

    if feature_user_dict is None:
        feature_user_dict[feature] = {}

    if feature_user_dict[feature][user.name] is None:
        feature_means = user
        feature_means[feature] = movies[feature]
        feature_user_dict[feature][user.name] = feature_means.groupby(feature).mean()

    # movies['mean_year'] = means_per_year.groupby('year').mean()
