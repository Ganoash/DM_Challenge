import numpy as np
import pandas as pd
import locality_sensitive_hashing


def similarity(x: pd.Series, y: pd.Series, distances: dict, column_means: pd.DataFrame):
    x_normalized = x - column_means[x.name]
    y_normalized = y - column_means[y.name]
    x_dist = 0
    y_dist = 0
    if x.name in distances:
        x_dist = distances[x.name]
    else:
        x_dist = euclidean_distance(x_normalized)
        distances[x.name] = x_dist

    if y.name in distances:
        y_dist = distances[y.name]
    else:
        y_dist = euclidean_distance(y_normalized)
        distances[x.name] = y_dist

    ret_value = x_normalized.dot(y_normalized) / (x_dist * y_dist)
    if pd.isna(ret_value):
        return 0
    return ret_value


def euclidean_distance(x: pd.Series):
    total = 0
    for value in x:
        total += value ** 2
    return np.sqrt(total)


def predict(similar: pd.Series, column_means):
    rating = 0
    similarity_sum: int = similar.sum()

    if similarity_sum == 0:
        return 0

    for index, item in similar.iteritems():
        rating += item * \
                  column_means[index[1]]

    return rating / similarity_sum


def create_prediction_matrix(similarity_matrix: pd.DataFrame, column_means: pd.DataFrame, k: int):
    prediction_matrix: pd.DataFrame = pd.DataFrame(0, index=range(len(column_means)), columns=["rating"])
    print("started prediction matrix")
    for i in range(len(similarity_matrix.columns)):
        similarities: pd.DataFrame = similarity_matrix.iloc[i].to_frame()
        top_similarities: pd.DataFrame = similarities.stack().nlargest(k)
        prediction = predict(top_similarities, column_means)
        prediction_matrix.loc[i+1, "rating"] = prediction
    return prediction_matrix


def create_similarity_matrix(utility: pd.DataFrame):
    column_means = utility.mean(axis=1).fillna(0)
    utility = utility.fillna(0)
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(utility, utility.T)

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
    cosine = pd.DataFrame(cosine, index=range(1,6041), columns=range(1,6041))
    return cosine, column_means

def lsh(utility: pd.DataFrame):
    return locality_sensitive_hashing.run(utility), utility.mean(axis=1).fillna(0)

def run(utility_matrix):
    return create_prediction_matrix(*create_similarity_matrix(utility_matrix), 10)
