import pandas as pd
import main

# function calculating the average rating per user
# returns a column with in each row the bias of each user
def user_bias():
    utility = main.utility_matrix - overall_bias()
    return pd.DataFrame(utility.mean(axis=1))


# function calculating the average value per movie
# returns a row with in each column the bias of each movie
def movie_bias():
    utility = main.utility_matrix - overall_bias()
    return pd.DataFrame(utility.mean(axis=0)).T


# function calculating the average value of all ratings
# returns an integer with the average value of ratings
def overall_bias():
    return main.ratings_description["rating"].mean()