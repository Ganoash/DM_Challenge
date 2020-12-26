import pandas as pd
from main import utility_matrix, ratings_description


# function calculating the average rating per user
# returns a column with in each row the bias of each user
def user_bias():
    utility = utility_matrix - overall_bias()
    return pd.DataFrame(utility.mean(axis=1))


# function calculating the average value per movie
# returns a row with in each column the bias of each movie
def movie_bias():
    utility = utility_matrix - overall_bias()
    return pd.DataFrame(utility.mean(axis=0)).T


# function calculating the average value of all ratings
# returns an integer with the average value of ratings
def overall_bias():
    return ratings_description["rating"].mean()


user = user_bias()
movie = movie_bias()

print(user_bias())
print(movie_bias())
