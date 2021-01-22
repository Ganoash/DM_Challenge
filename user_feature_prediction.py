from typing import Dict

import pandas as pd
import numpy as np

feature_dict: Dict[str, pd.Series] = {}
feature_movie_dict: Dict[str, pd.Series] = {}


def feature_bias(users: pd.DataFrame, utility: pd.DataFrame, feature: str):
    # take collumn mean and allign movies index to utility index
    feature_means = pd.DataFrame(utility.mean(axis=1))
    # feature_means = feature_means.reindex(sorted(feature_means.index), axis=1)
    # match collumn mean with feature
    feature_means[feature] = users[feature]

    # group by feature and take the mean value per feature
    feature_dict[feature] = feature_means.groupby(feature).mean()

    return feature_dict[feature]


def item_bias_matrix(users: pd.DataFrame, utility: pd.DataFrame):
    features = ["gender", "age", "profession"]
    feature_bias_users = {}
    for feature in features:
        feature_bias_users[feature] = feature_bias(users, utility, feature)

    average_feature_bias_per_user = {}
    for key, item in feature_bias_users.items():
        average_feature_bias_per_user[key] = pd.DataFrame(feature_bias_users[key].loc[users.loc[:, key]].to_numpy(), index=utility.index)

    print(average_feature_bias_per_user["gender"])

    feature_deviation_bias = {}
    for feature in features:
        feature_deviation_bias[feature] = average_feature_bias_per_user[feature][0] - utility.mean(axis=1).fillna(0)

    print(feature_deviation_bias)

    for feature in features:
        np.savetxt(feature + "_average_per_user.csv", average_feature_bias_per_user[feature].to_numpy())
        np.savetxt(feature + "_devation_per_user.csv", feature_deviation_bias[feature].to_numpy())

    #feature_deviation_bias = average_year_bias_per_movie[0].sub(utility.mean(axis=0)).fillna(0)

    #return (average_year_bias_per_movie, feature_deviation_bias)
