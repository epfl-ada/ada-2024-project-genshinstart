import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
import pandas as pd
import random
import plotly.graph_objects as go

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

def calculate_degree_condition_on_low_similarity_with_dest(path, graph, embeddings, threshold=0.85):
    # Calculate the rank of degree of chosen nodes if the similarity with the destination is low
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
        neighbors_similarity = [embedding_distance(node, destination, embeddings) for node in neighbors]

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