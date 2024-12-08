import numpy as np

def calculate_shortest_path_length(path, shortest_path_matrix):
    articles = path.split(';')
    start_article = articles[0]
    end_article = articles[-1]
    if start_article != end_article:
        try:
            return shortest_path_matrix[start_article][end_article]
        except KeyError:
            return np.nan
    return 0
