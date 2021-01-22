import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: Shirani
SURNAME: Bisnajak
STUDENT ID: 4440250
KAGGLE ID: Shirani Bisnajak

NAME: Lucas
SURNAME: Kroes
STUDENT ID: 4823257
KAGGLE ID: Lucas Kroes

### NOTES
This files is an example of what your code should look like. 
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/final-submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])


#####
##
# COLLABORATIVE FILTERING
##
#####
def predict_collaborative_filtering(utility):
    return run(utility, predictions_description)


def predict(movies, users, ratings, predictions):
    """
    The final prediction is based on LF model which is trained in the "Final Latent Factors Model (SGD + Regularized + bias).py" file
    Here we simply load in the artifacts for the the Q, P and biases to do the predictions.
    """
    from numpy import genfromtxt
    Q = genfromtxt("./lf_bias/Q.csv")
    P = genfromtxt("./lf_bias/P.csv")
    overall_bias = ratings['rating'].mean()
    user_bias = genfromtxt("./lf_bias/userbias.csv")
    movie_bias = genfromtxt("./lf_bias/moviebias.csv")
    submission = []
    R_pred = (
        overall_bias
        + np.tile(user_bias, (len(movies), 1))
        + np.tile(movie_bias, (len(users), 1)).T
        + np.matmul(Q, P)
    )
    for i, [user, movie] in enumerate(predictions_description.values):
        submission.append([i + 1, R_pred[movie - 1, user - 1]])
    return submission


if __name__ == "__main__":
    #####
    ##
    ## SAVE RESULTS
    ##
    #####    
    # //!!\\ TO CHANGE by your prediction function
    predictions = predict(movies_description, users_description, ratings_description, predictions_description)

    #Save predictions, should be in the form 'list of tuples' or 'list of lists'
    with open(submission_file, 'w') as submission_writer:
        #Formates data
        predictions = [map(str, row) for row in predictions]
        predictions = [','.join(row) for row in predictions]
        predictions = 'Id,Rating\n'+'\n'.join(predictions)
        
        #Writes it dowmn
        submission_writer.write(predictions)