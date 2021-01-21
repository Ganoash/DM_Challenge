from typing import Dict

import pandas as pd
import numpy as np

feature_dict: Dict[str, pd.Series] = {}
feature_user_dict: Dict[str, pd.Series] = {}


def feature_bias(movies: pd.DataFrame, utility: pd.DataFrame, feature: str):
    if feature in feature_dict:
        return feature_dict[feature]
    # take collumn mean and allign movies index to utility index
    feature_means = pd.DataFrame(utility.mean(axis=0))
    feature_means = feature_means.reindex(sorted(feature_means.index), axis=0)
    # match collumn mean with feature
    feature_means[feature] = movies[feature]

    # group by feature and take the mean value per feature
    feature_dict[feature] = feature_means.groupby(feature).mean()
    feature_dict[feature] = feature_dict[feature].reindex(feature_dict[feature].index.astype(int))

    return feature_dict[feature]


def item_bias_matrix(movies: pd.DataFrame, utility: pd.DataFrame):
    feature_bias_movies = feature_bias(movies, utility, "year")


    average_year_bias_per_movie = pd.DataFrame(feature_bias_movies.loc[movies.loc[:, "year"]].to_numpy(), index=utility.columns)
    feature_deviation_bias = average_year_bias_per_movie[0] - utility.mean(axis=0).fillna(0)

    np.savetxt("year_average_per_movie.csv", average_year_bias_per_movie.to_numpy())
    np.savetxt("year_deviation_per_movie.csv", feature_deviation_bias.to_numpy())