import numpy as np
import plotly.graph_objects as go
import random
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.evaluation_utils import embedding_distance
import seaborn as sns
import pandas as pd


def plot_updated_task_path_length_comparison(before_tuning_paths, after_4o_tuning_paths, after_4o_mini_tuning_paths, ori_dest, shortest_paths, sample_fraction=0.15, fixed_height_per_pair=50, fixed_width=1400, x_jitter_offset=0.025, seed=58, y_offset=0.15):
    before_tuning_path_lens = []
    after_4o_path_lens = []
    after_4o_mini_path_lens = []
    shortest_path_lens = []
    success_pairs = []

    for pair in ori_dest:
        path_before_tuning = before_tuning_paths[pair]
        path_after_4o_tuning = after_4o_tuning_paths[pair]
        path_after_4o_mini_tuning = after_4o_mini_tuning_paths[pair]
        path_shortest = shortest_paths[pair]

        before_tuning_path_len = [len(path) for path in path_before_tuning if path is not None]
        after_4o_path_len = [len(path) for path in path_after_4o_tuning if path is not None]
        after_4o_mini_path_len = [len(path) for path in path_after_4o_mini_tuning if path is not None]

        if before_tuning_path_len and after_4o_path_len and after_4o_mini_path_len:
            before_tuning_path_lens.append(before_tuning_path_len)
            after_4o_path_lens.append(after_4o_path_len)
            after_4o_mini_path_lens.append(after_4o_mini_path_len)
            shortest_path_lens.append([len(path_shortest[0])])
            success_pairs.append(pair)
        
    mean_before_tuning_path_len = [np.mean(lengths) for lengths in before_tuning_path_lens]
    sorted_indices = np.argsort(mean_before_tuning_path_len)
    sorted_ori_dest = [success_pairs[i] for i in sorted_indices]
    sorted_before_tuning_path_len = [before_tuning_path_lens[i] for i in sorted_indices]
    sorted_after_4o_path_len = [after_4o_path_lens[i] for i in sorted_indices]
    sorted_after_4o_mini_path_len = [after_4o_mini_path_lens[i] for i in sorted_indices]
    sorted_shortest_path_len = [shortest_path_lens[i] for i in sorted_indices]
    
    np.random.seed(seed)

    # Standard Colors
    DEEP_SKY_BLUE = "rgba(0, 191, 255, 0.7)"  # Before tuning
    CORAL = "rgba(255, 127, 80, 0.7)"         # After 4o tuning
    MEDIUM_SEA_GREEN = "rgba(60, 179, 113, 0.7)"  # After 4o mini tuning
    GOLDENROD = "rgba(218, 165, 32, 0.7)"      # Shortest path

    BACKGROUND = "#F0F4FF"                   # Plot background
    LIGHT_GRAY = "#E0E8F0"                   # Grid lines

    # Sample indices first and preserve original order
    def sample_indices(data, fraction):
        total = len(data)
        num_samples = max(1, int(total * fraction))
        sampled_indices = sorted(np.random.choice(total, num_samples, replace=False))
        return sampled_indices

    # Sample the indices
    sampled_indices = sample_indices(sorted_before_tuning_path_len, sample_fraction)

    # Assign y-positions for alignment
    y_positions = np.arange(len(sampled_indices))

    # Adjust figure height
    total_height = fixed_height_per_pair * len(y_positions)

    # Create the Plotly figure
    fig = go.Figure()

    # Add grey reference lines for each pair
    for y in y_positions:
        fig.add_shape(
            type="line",
            x0=0, x1=1, y0=y, y1=y,
            xref="paper", yref="y",
            line=dict(color=LIGHT_GRAY, width=1, dash="dot")
        )

    # Plot Shortest paths
    for y, lengths in zip(y_positions, sorted_shortest_path_len):
        if lengths:
            fig.add_trace(go.Scatter(
                x=lengths,
                y=[y] * len(lengths),
                mode='markers',
                name="Shortest",
                marker=dict(color=GOLDENROD, size=10, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>Shortest Path Length:</b> {lengths[0]}<extra></extra>",
                showlegend=bool(y == 0)
            ))
    
    # Plot Before Tuning paths based on the mean path length
    mean_before_tuning_path_len = [np.mean(lengths) for lengths in sorted_before_tuning_path_len]
    for y, length in zip(y_positions, mean_before_tuning_path_len):
        fig.add_trace(go.Scatter(
            x=[length],
            y=[y + y_offset],
            mode='markers',
            name="Before Tuning",
            marker=dict(color=DEEP_SKY_BLUE, size=10, line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>Mean Path Length:</b> {length}<extra></extra>",
            showlegend=bool(y == 0)
        ))
    
    # Plot After 4o Tuning paths based on the mean path length
    mean_after_4o_path_len = [np.mean(lengths) for lengths in sorted_after_4o_path_len]
    for y, length in zip(y_positions, mean_after_4o_path_len):
        fig.add_trace(go.Scatter(
            x=[length],
            y=[y],
            mode='markers',
            name="After 4o Tuning",
            marker=dict(color=CORAL, size=10, line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>Mean Path Length:</b> {length}<extra></extra>",
            showlegend=bool(y == 0)
        ))

    # Plot After 4o Mini Tuning paths based on the mean path length
    mean_after_4o_mini_path_len = [np.mean(lengths) for lengths in sorted_after_4o_mini_path_len]
    for y, length in zip(y_positions, mean_after_4o_mini_path_len):
        fig.add_trace(go.Scatter(
            x=[length],
            y=[y - y_offset],
            mode='markers',
            name="After 4o Mini Tuning",
            marker=dict(color=MEDIUM_SEA_GREEN, size=10, line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>Mean Path Length:</b> {length}<extra></extra>",
            showlegend=bool(y == 0)
        ))
        
    # Update layout
    fig.update_layout(
        title=dict(
            text="Path Length Comparison For Before and After Improvement Paths",
            font=dict(family="Arial, sans-serif", size=24, color="rgba(0, 0, 0, 0.7)")
        ),
        xaxis=dict(
            title="Path Length",
            title_standoff=10,  # Add space between title and axis
            side="top",  # Move the title to the top
            gridcolor=LIGHT_GRAY,
            zerolinecolor=LIGHT_GRAY
        ),
        yaxis=dict(
            tickvals=y_positions,
            ticktext=[f"{pair[0]} → {pair[1]}" for pair in sorted_ori_dest],
            tickangle=0,
            tickfont=dict(size=15, family="Arial, sans-serif", color="rgba(0, 0, 0, 0.8)"),
            gridcolor=LIGHT_GRAY
        ),
        plot_bgcolor=BACKGROUND,
        paper_bgcolor="white",
        width=fixed_width,
        height=total_height,
        showlegend=True,
        legend=dict(
            font=dict(color="rgba(0, 0, 0, 0.7)"),
            bgcolor="white",
            bordercolor=LIGHT_GRAY,
            borderwidth=1
        )
    )

    # Show the plot
    fig.show()

def correlation_between_embedding_and_distance(graph, embeddings, sample_num):
    # sample from pair from the graph, and calculate the distance between the pair, as well as the cosine similarity between the pair
    distances = []
    cosine_distances = []
    for _ in range(sample_num):
        node1 = random.choice(list(graph.nodes))
        node2 = random.choice(list(graph.nodes))
        if not nx.has_path(graph, node1, node2):
            continue
        distance = nx.shortest_path_length(graph, node1, node2)
        cosine_distance = embedding_distance(node1, node2, embeddings)
        distances.append(distance)
        cosine_distances.append(cosine_distance)
    
    # plot to show the correlation between the distance and the similarity 
    # use seaborn to plot more beautiful plot
    '''
    plt.scatter(distances, cosine_distances)
    plt.xlabel("Distance")
    plt.ylabel("Cosine Distance")
    plt.title("Correlation between Distance and Embedding Similarity")
    plt.show()
    '''
    
    df = pd.DataFrame({'distance': distances, 'cosine distance': cosine_distances})
    sns.jointplot(data=df, x='distance', y='cosine_distance', kind='reg')
    plt.show()
