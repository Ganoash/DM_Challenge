import numpy as np
import pandas as pd
import locality_sensitive_hashing

"""" 
inputs a matrix of similarity values and means per item, outputs the element-wise product of similarity and 
means summed and normalized
"""


def predict(similar: pd.Series, means):
    similarity_sum: int = similar.sum()
    # reduce means to necessary indexes
    means = means.loc[similar.index]

    # take the elementwise product of the vectors and sum
    rating = (similar.values.dot(means.values)).sum()

    if similarity_sum == 0:
        return 0

    return rating / similarity_sum


"""
takes a user-user similarity matrix and a item-item similarity matrix, along with a utility matrix and the prediction file,
outputs the prediction in a list format so it can be written to a file.
"""


def create_prediction_matrices(item_similarity: pd.DataFrame, user_similarity: pd.DataFrame, utility: pd.DataFrame,
                               k: int, prediction_file: pd.DataFrame):
    # create prediction matrices from similarity matrices
    item_prediction = create_prediction_matrix(item_similarity, utility.mean(axis=0).fillna(0), k)
    user_prediction = create_prediction_matrix(user_similarity, utility.mean(axis=1).fillna(0), k)

    # l1 based on rough calculation on the dataset. System is non-linear so no gradient descent possible
    l1 = 0.65

    # mapping to dictionaries, taking unnecesary steps and code could be improved massively
    item_dict = {}
    user_dict = {}
    for index, item in enumerate(item_prediction.values):
        item_dict[index + 1] = l1 * item[0]

    for index, user in enumerate(user_prediction.values):
        user_dict[index + 1] = (1 - l1) * user[0]

    # take element-wise addition of items and users
    ret_value = pd.DataFrame(prediction_file["movieID"]).applymap(lambda x: item_dict[x]).to_numpy() + pd.DataFrame(
        prediction_file["userID"]).applymap(lambda x: user_dict[x]).to_numpy()

    # parse table to output format
    ret_value = pd.DataFrame(ret_value, columns=["movieID"])

    print(ret_value)
    ret_value["index"] = ret_value.index
    ret_index = (ret_value["index"] + 1).values.astype(int)
    ret_value = ret_value["movieID"].values

    return zip(ret_index, ret_value)


"""
creates prediction matrix based on a similarity matrix and a list of means. K value indicates how many similar items you want to consider
"""


def create_prediction_matrix(similarity_matrix: pd.DataFrame, means: pd.DataFrame, k: int):
    prediction_matrix: pd.DataFrame = pd.DataFrame(0, index=range(1, len(means)), columns=["rating"])
    print("started prediction matrix")
    for i in range(len(means)):
        similarities: pd.Series = similarity_matrix.iloc[i]
        top_similarities: pd.Series = similarities.nlargest(k)
        prediction = predict(top_similarities, means)
        prediction_matrix.loc[i + 1, "rating"] = prediction
    return prediction_matrix


"""
outputs the cosine similarity between all rows
"""


def create_similarity_matrix(utility: pd.DataFrame):
    #subtract row means
    utility = (utility - utility.mean(axis=0)).fillna(0)

    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(utility.T, utility)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    cosine = pd.DataFrame(cosine, index=range(1, len(utility.columns) + 1), columns=range(1, len(utility.columns) + 1))
    return cosine


"""
calculate cosine similarity through minhashing
"""


def lsh(utility: pd.DataFrame):
    return locality_sensitive_hashing.run(utility)


def run(utility_matrix, predictions):
    return create_prediction_matrices(create_similarity_matrix(utility_matrix),
                                      create_similarity_matrix(utility_matrix.T), utility_matrix, 5, predictions)

