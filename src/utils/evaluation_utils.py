import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine

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

def get_shortest_path(G, ori_dest):
    shortest_paths = {}
    for pair in ori_dest:
        start_article, target_article = pair
        try:
            shortest_path = nx.shortest_path(G, start_article, target_article)
            shortest_paths[pair] = [shortest_path]
        except nx.NetworkXNoPath:
            shortest_paths[pair] = []
    return shortest_paths

def embedding_distance(node1, node2, embeddings):
    if node1 not in embeddings or node2 not in embeddings:
        return np.nan  
    return cosine(embeddings[node1], embeddings[node2])

def calculate_closeness_scores(path, embeddings):
    articles = path
    destination = articles[-1]  
    closeness_scores = []

    nodes_stack = []
    prev_node = articles[0]

    for i in range(1, len(articles)):
        current_node = articles[i]
        if current_node == '<' or current_node == 'back':
            try:
                current_node = nodes_stack.pop()
            except IndexError:
                raise ValueError("Back in the beginning")
        else:
            nodes_stack.append(prev_node)

        # Calculate the distance from the current node to the destination
        current_distance = embedding_distance(current_node, destination, embeddings)
        prev_distance = embedding_distance(prev_node, destination, embeddings)
        
        # Calculate the closeness score: distance difference from previous move
        closeness_score = prev_distance - current_distance
        closeness_scores.append(closeness_score)

        # Update the previous distance for the next move
        prev_node = current_node
        
    return closeness_scores

def rank_neighbors(previous_node, current_node, destination, graph, embeddings):
    # rank the neighbors including the previous node of the current node based on the distance to the destination
    neighbors = list(graph.successors(current_node))
    if previous_node is not None and previous_node not in neighbors:
        neighbors.append(previous_node)
    distance_to_destination = [embedding_distance(node, destination, embeddings) for node in neighbors]
    sorted_neighbors = [x for _, x in sorted(zip(distance_to_destination, neighbors))]
    return sorted_neighbors

def calculate_rank_of_neighbors(path, graph, embeddings):
    articles = path
    destination = articles[-1]  
    ranks = []

    nodes_stack = []
    prev_node = articles[0]

    for i in range(1, len(articles)):
        if prev_node not in graph.nodes:
            print(f"Node {prev_node} is not in the graph")
            return []
        if nodes_stack.__len__() == 0:
            neighbors = rank_neighbors(None, prev_node, destination, graph, embeddings)
        else:
            neighbors = rank_neighbors(nodes_stack[-1], prev_node, destination, graph, embeddings)
        current_node = articles[i]
        if current_node == '<' or current_node == 'back':
            try:
                current_node = nodes_stack.pop()
            except IndexError:
                raise ValueError("Back in the beginning")
        else:
            nodes_stack.append(prev_node)

        # Rank the neighbors of the current node
        if current_node in neighbors:
            rank = neighbors.index(current_node) + 1
            ranks.append(rank)
        else:
            print(f"Node {current_node} is not in the neighbors of {prev_node}")

        # Update the previous distance for the next move
        prev_node = current_node
        
    return ranks

