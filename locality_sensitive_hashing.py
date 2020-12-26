import pandas as pd
import numpy as np


def create_normal(d : int):
        normal = np.random.randn(d)
        normal /= np.linalg.norm(normal)
        return normal.tolist()


def lsh(utility: pd.DataFrame):
    np.random.seed(42)
    hyperplanes = generate_hyperplanes(len(utility.columns), 1024)
    utility = utility.fillna(0)
    signature_matrix = pd.DataFrame(hyperplanes.dot(utility.T))

    similarity_matrix = generate_similarity_matrix(signature_matrix)
    print(similarity_matrix)
    return similarity_matrix


def generate_hyperplanes(d: int, n: int):
    hyperplanes = []
    for i in range(n):
        hyperplanes.append(create_normal(d))
    return np.array(hyperplanes)


def generate_signature_vector(hyperplanes: list, vector: np.array):
    return [hyperplane.side(vector) for hyperplane in hyperplanes]


def generate_similarity_matrix(signature_matrix: pd.DataFrame):
    similarity_matrix = pd.DataFrame(np.nan, index=range(1, len(signature_matrix.columns)), columns=range(1, len(signature_matrix.columns)))
    for i in range(1, len(signature_matrix.columns)):
        print(i)
        for j in range(1, i):
            similarity_matrix.loc[i, j] = similarity(signature_matrix.iloc[i], signature_matrix.iloc[j])
    return similarity_matrix


def similarity(a: pd.Series, b: pd.Series):
    return np.cos(np.pi * (1 - len(a[a == b]) / len(a)))


def run(utility: pd.DataFrame):
    similarity_matrix = lsh(utility)
    return similarity_matrix
