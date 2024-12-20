import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
import pandas as pd
import random
import plotly.graph_objects as go
from scipy.stats import ttest_rel

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
    '''
    Get the shortest path for each pair of (start article, target article)
    Parameters:
    - G: nx.DiGraph, the graph
    - ori_dest: list, the list of (start article, target article) pairs

    Returns:
    - shortest_paths: dict, the shortest paths
    '''
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
    '''
    Calculate the cosine distance between two nodes
    Parameters:
    - node1: str, the first node
    - node2: str, the second node
    - embeddings: dict, the embeddings
    
    Returns:
    - float, the cosine distance of the two nodes
    '''
    if node1 not in embeddings or node2 not in embeddings:
        return np.nan  
    return cosine(embeddings[node1], embeddings[node2])

def calculate_closeness_scores(path, embeddings):
    '''
    Calculate the closeness scores for each move in the path
    The closeness score is the difference between the previous move's distance and the current move's distance to the destination
    Parameters:
    - path: str, the path
    - embeddings: dict, the embeddings

    Returns:
    - closeness_scores: list, the closeness scores for each move
    '''
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
    '''
    Rank the neighbors of the current node based on the distance to the destination
    Parameters:
    - previous_node: str, the previous node
    - current_node: str, the current node
    - destination: str, the destination node
    - graph: nx.Graph, the graph
    - embeddings: dict, the embeddings
    
    Returns:
    - sorted_neighbors: list, the sorted neighbors of the current node
    '''
    neighbors = list(graph.successors(current_node))
    if previous_node is not None and previous_node not in neighbors:
        neighbors.append(previous_node)
    distance_to_destination = [embedding_distance(node, destination, embeddings) for node in neighbors]
    sorted_neighbors = [x for _, x in sorted(zip(distance_to_destination, neighbors))]
    return sorted_neighbors

def calculate_rank_of_neighbors(path, graph, embeddings):
    '''
    Calculate the rank of the chosen nodes in the neighbors of the previous node for each move in the path
    Parameters:
    - path: str, the path
    - graph: nx.Graph, the graph
    - embeddings: dict, the embeddings
    
    Returns:
    - ranks: list, the ranks of the chosen nodes
    '''
    articles = path
    destination = articles[-1]  
    ranks = []

    nodes_stack = []
    prev_node = articles[0]

    for i in range(1, len(articles)):
        if prev_node not in graph.nodes:
            print(f"Node {prev_node} is not in the graph")
            return []
        
        # Get the neighbors of the previous node and rank them based on the distance to the destination
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

def process_paths_to_categories(paths, categories_df):
    nodes = {}
    article_to_category = dict(zip(categories_df['article'], categories_df['category']))

    # store the messages
    all_articles = set()
    for path_list in paths.values():
        for path in path_list:
            all_articles.update(path)

    for article in all_articles:
        category_parts = article_to_category.get(article, 'Unknown').split('.')
        path_parts = category_parts + [article]

        for i, part in enumerate(path_parts):
            node_id = '.'.join(category_parts[:i] + [part])
            parent_id = '.'.join(category_parts[:i]) if i > 0 else ''
            if node_id not in nodes:
                nodes[node_id] = {
                    'id': node_id,
                    'name': part,
                    'parent': parent_id,
                    'size': 0
                }
            nodes[node_id]['size'] += 1

    # build DataFrame
    nodes_df = pd.DataFrame.from_dict(nodes, orient='index')

    parent_sizes = nodes_df.set_index('id')['size'].to_dict()
    nodes_df['relative_size'] = nodes_df.apply(
        lambda row: (row['size'] / parent_sizes.get(row['parent'], row['size'])) * 100
        if parent_sizes.get(row['parent'], row['size']) else 100,
        axis=1
    )
    return nodes_df

def create_treemap(ids, labels, parents, values, hover_text, title):
    colors = [f"#{random.randint(30, 150):02x}{random.randint(30, 150):02x}{random.randint(30, 150):02x}" for _ in ids]

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        hovertext=hover_text,
        hoverinfo="text",
        maxdepth=2,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(width=1, color='#000000')),
        textfont=dict(color='white', size=16),
        textposition="middle center",
        hoverlabel=dict(bgcolor='#2c3e50', font=dict(color='white'))
    ))

    fig.update_layout(
        width=1000,
        height=800,
        title=dict(
            text=title,
            xanchor='center',
            x=0.5,
            yanchor='top',
            font=dict(color='white')
        ),
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        font=dict(color='white')
    )
    return fig

def visualize_paths_treemap(paths, categories_df, title):
    nodes_df = process_paths_to_categories(paths, categories_df)

    # print(f"Category Statistics for {title}:")
    # print(nodes_df[['name', 'size', 'relative_size']])

    # Visualization
    ids = nodes_df['id']
    labels = nodes_df.apply(
        lambda row: f"{row['name']}" if row['size'] == 1 else f"{row['name']}<br>({row['relative_size']:.2f}%)",
        axis=1
    )
    parents = nodes_df['parent']
    values = nodes_df['size']

    hover_text = nodes_df.apply(
        lambda row: "Article" if row['size'] == 1 else
        f"Full Category: {row['id']}<br>"
        f"Size: {row['size']}<br>"
        f"Percentage: {row['relative_size']:.2f}%",
        axis=1
    )

    fig = create_treemap(ids, labels, parents, values, hover_text, title)
    fig.show()

    return nodes_df



def calculate_target_scores(path, embeddings):
    articles = path
    destination = articles[-1]  
    target_scores = []

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
        current_distance = 1 - embedding_distance(current_node, destination, embeddings)
        target_scores.append(current_distance)

        # Update the previous distance for the next move
        prev_node = current_node
        
    return target_scores


def score_filtering(closeness_scores, path_len, len, first_n):
    filtered_scores = {}
    for (key, scores), lengths in zip(closeness_scores.items(), path_len):

        for length, scores in zip(lengths, scores):

            if length >= len:
                filtered_scores.setdefault(key, []).append(scores[:first_n])
    return filtered_scores

# Node Degree Comparison
def compute_path_degrees(paths, degrees):
    paths_degrees = {
        key: [[degrees.get(node, 0) for node in path] for path in path_list]
        for key, path_list in paths.items()
    }
    return paths_degrees

def calculate_degree_condition_on_low_similarity_with_dest(path, graph, embeddings, threshold=0.30):
    '''
    Calculate the rank of the chosen nodes in the neighbors of the previous node for each move in the path, conditioned on low similarity with the destination
    Parameters:
    - path: str, the path
    - graph: nx.Graph, the graph
    - embeddings: dict, the embeddings
    - threshold: float, the threshold for low similarity
    
    Returns:
    - rank_of_degrees: list, the ranks of the chosen neighbors for each move
    '''
    articles = path
    destination = articles[-1]
    rank_of_degrees = []

    nodes_stack = []
    prev_node = articles[0]

    if '<' in articles or 'back' in articles:
        return []

    for i in range(1, len(articles)):
        if prev_node not in graph.nodes:
            print(f"Node {prev_node} is not in the graph")
            return []
        neighbors = list(graph.successors(prev_node))
        neighbors_similarity = [1 - embedding_distance(node, destination, embeddings) for node in neighbors]

        current_node = articles[i]
        if current_node == destination:
            break

        if all(similarity < threshold for similarity in neighbors_similarity):
            degrees = [graph.degree(node) for node in neighbors]
            if current_node in neighbors:
                rank = sorted(degrees).index(graph.degree(current_node)) + 1
                rank_of_degrees.append(rank)
            else:
                print(f"Node {current_node} is not in the neighbors of {prev_node}")
        
        prev_node = current_node
    
    return rank_of_degrees

def paths_comparation(paths_1, paths_2, shortest_path, ori_dest):
    '''
    Compare the two paths and calculate the statistics
    Parameters:
    - paths_1: dict, the paths for the first method
    - paths_2: dict, the paths for the second method
    - shortest_path: dict, the shortest paths
    - ori_dest: list, the list of (start article, target article) pairs

    Returns:
    - None

    Output:
    - Print the statistics
    '''
    failure_1 = 0
    failure_2 = 0
    shorter_1 = 0
    shorter_2 = 0
    paths_length_1 = []
    paths_length_2 = []
    difference_with_shortest_1 = []
    difference_with_shortest_2 = []

    for pair in ori_dest:
        start_article, target_article = pair
        paths = paths_1[pair]
        path_lengths = [len(path) for path in paths if path is not None]
        if len(path_lengths) == 0:
            avg_path_length_1 = 'No path found'
        else:
            avg_path_length_1 = np.mean(path_lengths)
        if avg_path_length_1 == 'No path found':
            failure_1 += 1

        paths = paths_2[pair]
        path_lengths = [len(path) for path in paths if path is not None]
        if len(path_lengths) == 0:
            avg_path_length_2 = 'No path found'
        else:
            avg_path_length_2 = np.mean(path_lengths)
        if avg_path_length_2 == 'No path found':
            failure_2 += 1

        if avg_path_length_1 == 'No path found' or avg_path_length_2 == 'No path found':
            continue
        
        if avg_path_length_1 > avg_path_length_2:
            shorter_2 += 1
        elif avg_path_length_1 < avg_path_length_2:
            shorter_1 += 1
        
        paths_length_1.append(avg_path_length_1)
        paths_length_2.append(avg_path_length_2)
        difference_with_shortest_1.append(avg_path_length_1 - len(shortest_path[pair][0]))
        difference_with_shortest_2.append(avg_path_length_2 - len(shortest_path[pair][0]))

        
    print(f'Failure rate for Paths 1: {failure_1/len(ori_dest)}, Path 2: {failure_2/len(ori_dest)}')
    print(f'Number of paths that are shorter for Paths 1: {shorter_1}, Path 2: {shorter_2}')

    # calculate the statistics of the path length
    paths_length_1 = [length for length in paths_length_1 if length != 'No path found']
    paths_length_2 = [length for length in paths_length_2 if length != 'No path found']
    print(f'Average for Path 1: {np.mean(paths_length_1)}, Path 2: {np.mean(paths_length_2)}')
    print(f'Median path length for Path 1: {np.median(paths_length_1)}, Path 2: {np.median(paths_length_2)}')
    # t test
    t_stat, p_value = ttest_rel(paths_length_1, paths_length_2)
    print(f'T test result: t_stat={t_stat}, p_value={p_value}')

    # calculate the statistics of the difference with the shortest path
    print(f'Average difference with the shortest path for Path 1: {np.mean(difference_with_shortest_1)}, Path 2: {np.mean(difference_with_shortest_2)}')

    # t test
    t_stat, p_value = ttest_rel(difference_with_shortest_1, difference_with_shortest_2)
    print(f'T test result: t_stat={t_stat}, p_value={p_value}')