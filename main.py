import pandas as pd
import numpy as np
data_location: str = "Data/"

movies_filename: str = "movies.csv"
ratings_filename: str = "ratings.csv"
users_filename: str = "users.csv"
predictions_filename: str = "predictions.csv"
submissions_filename: str = "submissions.csv"


def create_utility_matrix(rating_location: str):
    rating: pd.DataFrame = pd.read_csv(rating_location, sep=';', names=["user", "movie", "rating"])

    util: pd.DataFrame = rating.pivot(index="user", columns="movie", values="rating").fillna(0).loc[1:, 1:]

    print(util)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    create_utility_matrix("".join((data_location, ratings_filename)))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
