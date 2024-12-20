import numpy as np
import plotly.graph_objects as go
import random
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.evaluation_utils import embedding_distance
import seaborn as sns
import pandas as pd

# Use the same color definitions as in the notebook
PRIMARY_BLUE = "rgba(58, 85, 209, 0.7)"
SOFT_ORANGE = "rgba(242, 153, 74, 0.7)"
LIGHT_GREEN = "rgba(111, 207, 151, 0.8)"
BACKGROUND = "#F0F4FF"
LIGHT_GRAY = "#E0E8F0"


def plot_path_len_comparison(sampled_indices, aligned_ai_path_len, aligned_human_path_len, aligned_shortest_path_len, selected_ori_dest):
    # Move the plotting code for path length comparison here from the notebook
    # The code is originally defined in your original notebook for plot_path_len_comparison
    # Keep the logic the same.
    # ...
    pass


def plot_closeness_scores_distribution(ai_x, ai_y, human_x, human_y, optimal_x, optimal_y):
    # Move the code that creates the closeness scores distribution figure here
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ai_x, y=ai_y, mode='lines', name='AI path',
                             line=dict(color=PRIMARY_BLUE, width=3),
                             hovertemplate="AI path: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=human_x, y=human_y, mode='lines', name='Human path',
                             line=dict(color=SOFT_ORANGE, width=3),
                             hovertemplate="Human path: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=optimal_x, y=optimal_y, mode='lines', name='Shortest path',
                             line=dict(color=LIGHT_GREEN, width=3),
                             hovertemplate="Shortest path: %{y:.3f}<extra></extra>"))
    fig.update_layout(
        title="Closeness Scores Distribution",
        xaxis_title="Closeness Score",
        yaxis_title="Density",
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        legend=dict(x=0.7, y=0.95, bgcolor=BACKGROUND, bordercolor=LIGHT_GRAY, borderwidth=1),
        font=dict(size=12),
        width=900,
        height=600
    )
    fig.update_yaxes(showgrid=True, gridcolor=LIGHT_GRAY)
    fig.show()


def plot_node_selection_preference(ai_percentages, human_percentages, optimal_percentages):
    # Define fixed colors with transparency
    PRIMARY_BLUE = "rgba(58, 85, 209, 0.7)"  # AI paths - Light Blue with opacity
    SOFT_ORANGE = "rgba(242, 153, 74, 0.7)"  # Human paths - Soft Orange with opacity
    LIGHT_GREEN = "rgba(111, 207, 151, 0.8)" # Shortest paths - Light Green with slight opacity

    # X-axis labels (first 5 ranks)
    ranks = list(range(1, 6))

    # Create a bar chart
    fig = go.Figure()

    # AI path distribution
    fig.add_trace(go.Bar(
        x=ranks,
        y=ai_percentages,
        name='AI path',
        marker=dict(color=PRIMARY_BLUE),
        text=[f"{val:.2f}%" for val in ai_percentages],  # Add percentage labels
        textposition='outside',
        hovertemplate="AI: %{y:.2f}%<extra></extra>"  # Custom hover template
    ))

    # Human path distribution
    fig.add_trace(go.Bar(
        x=ranks,
        y=human_percentages,
        name='Human path',
        marker=dict(color=SOFT_ORANGE),
        text=[f"{val:.2f}%" for val in human_percentages],  # Add percentage labels
        textposition='outside',
        hovertemplate="Human: %{y:.2f}%<extra></extra>"  # Custom hover template
    ))

    # Shortest path distribution
    fig.add_trace(go.Bar(
        x=ranks,
        y=optimal_percentages,
        name='Shortest path',
        marker=dict(color=LIGHT_GREEN),
        text=[f"{val:.2f}%" for val in optimal_percentages],  # Add percentage labels
        textposition='outside',
        hovertemplate="Shortest: %{y:.2f}%<extra></extra>"  # Custom hover template
    ))

    # Update layout for aesthetics
    fig.update_layout(
        title="Node Selection Preference by Semantic Proximity",
        xaxis_title="Rank of Neighbors according to Semantic Proximity",
        yaxis_title="Percentage (Relative to All Choices)",
        barmode='group',  # Grouped bar chart
        plot_bgcolor="rgba(240, 244, 255, 1)",  # Background
        paper_bgcolor="rgba(240, 244, 255, 1)",  # Paper background
        font=dict(size=12),
        width=800,   # Fixed width
        height=500   # Fixed height
    )

    # Update grid line styles
    fig.update_xaxes(showgrid=True, gridcolor="rgba(224, 232, 240, 1)", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(224, 232, 240, 1)")

    # Show the interactive plot
    fig.show()
    # pio.write_html(fig, "pages/node_selection_preference_by_semantic_proximity.html", full_html=True)

    pass


def plot_accumulated_distribution_of_ranks(ai_scores_accumulated, human_scores_accumulated, optimal_scores_accumulated, title, xaxis_title, yaxis_title):
    # Move the code for plotting the accumulated distribution of rank of neighbors here
    # This is used multiple times with slightly different data, consider making it generic
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(50)),
        y=ai_scores_accumulated[:50],
        mode='lines',
        name='AI path',
        line=dict(color=PRIMARY_BLUE, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(50)),
        y=human_scores_accumulated[:50],
        mode='lines',
        name='Human path',
        line=dict(color=SOFT_ORANGE, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(50)),
        y=optimal_scores_accumulated[:50],
        mode='lines',
        name='Shortest path',
        line=dict(color=LIGHT_GREEN, width=3)
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        legend=dict(
            x=0.7, y=0.1,
            bgcolor=BACKGROUND,
            bordercolor=LIGHT_GRAY,
            borderwidth=1
        ),
        font=dict(size=12),
        width=900,
        height=600
    )
    fig.update_xaxes(showgrid=True, gridcolor=LIGHT_GRAY, dtick=5)
    fig.update_yaxes(showgrid=True, gridcolor=LIGHT_GRAY)
    fig.show()


def plot_histogram_of_path_lengths(human_lengths_capped, ai_lengths_capped):
    # Create the histogram with capped lengths
    bins = list(range(1, 22)) + [22]  # Adding a bin for >20
    plt.figure(figsize=(10, 6))
    plt.hist(human_lengths_capped, bins=bins, alpha=0.5, label="Human Paths", color="blue", density=True)
    plt.hist(ai_lengths_capped, bins=bins, alpha=0.5, label="AI Paths", color="orange", density=True)
    plt.xticks(range(1, 22), labels=[str(i) for i in range(1, 21)] + ['>20'])
    plt.title('Density Histogram of Path Lengths', fontsize=14)
    plt.xlabel('Path Length', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.show()


def correlation_between_embedding_and_distance(graph, embeddings, sample_num):
    '''
    Sample from pair from the graph, and calculate the distance between the pair, plot to show the correlation between the distance and the Cosine distance.
    Parameters:
    - graph: nx.Graph, the graph
    - embeddings: dict, the embeddings
    - sample_num: int, the sample number

    Returns:
    - None
    '''
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

    df = pd.DataFrame({'Distance': distances, 'Cosine Distance': cosine_distances})
    
    # Using the 'height' parameter directly:
    sns.jointplot(data=df, x='Distance', y='Cosine Distance', kind='reg', height=8)
    plt.show()



# Helper function to filter out large values within a sublist
def filter_outliers(data, threshold):
    mean = np.mean(data)
    return [x for x in data if x <= mean * threshold]

# Sample indices first and preserve original order
def sample_indices(data, fraction):
    total = len(data)
    num_samples = max(1, int(total * fraction))
    sampled_indices = sorted(np.random.choice(total, num_samples, replace=False))
    return sampled_indices

# Apply horizontal jitter for AI and Human paths
def jitter(data, offset):
    return [x + offset for x in data]

# Helper function to calculate min and max
def calculate_min_max(data):
    """Return min and max of a list."""
    if not data:  # Handle empty lists
        return None, None
    min_val = min(data)
    max_val = max(data)
    return min_val, max_val



def plot_path_len_comparison(sampled_indices, aligned_ai_path_len, aligned_human_path_len, aligned_shortest_path_len, selected_ori_dest):

    fixed_height_per_pair = 50  # Fixed height for each pair in pixels
    fixed_width = 1400  # Fixed width for the figure
    x_jitter_offset = 0.025  # Small horizontal shift to avoid overlapping
    seed = 58
    np.random.seed(seed)
    y_offset = 0.15

    # Assign y-positions for alignment
    y_positions = np.arange(len(sampled_indices))

    # Adjust figure height
    total_height = fixed_height_per_pair * len(y_positions)
    
    # Standard Colors with Transparency
    PRIMARY_BLUE = "rgba(58, 85, 209, 0.7)"  # AI paths - Light Blue with opacity
    SOFT_ORANGE = "rgba(242, 153, 74, 0.7)"  # Human paths - Soft Orange with opacity
    LIGHT_GREEN = "rgba(111, 207, 151, 0.8)" # Shortest paths - Light Green with slight opacity
    BACKGROUND = "#F0F4FF"                   # Plot background
    LIGHT_GRAY = "#E0E8F0"                   # Grid lines

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




    # Plot AI paths
    for y, lengths in zip(y_positions, aligned_ai_path_len):
        if lengths:
            min_val, max_val = calculate_min_max(lengths)
            median_val = np.median(lengths)  # Calculate median
            if min_val is not None:
                plot_max_val = max_val if max_val > min_val else min_val + 0.1
                fig.add_trace(go.Bar(
                    x=[plot_max_val - min_val],
                    y=[y + y_offset],
                    base=min_val,
                    name="AI",
                    marker=dict(color=PRIMARY_BLUE),
                    orientation='h',
                    width=0.35,
                    hovertemplate=f"<b>Min:</b> {min_val}<br>"
                                f"<b>Median:</b> {median_val}<br>"
                                f"<b>Max:</b> {max_val}<extra></extra>",
                    showlegend=bool(y == 0)
                ))

    # Plot Human paths
    for y, lengths in zip(y_positions, aligned_human_path_len):
        if lengths:
            min_val, max_val = calculate_min_max(lengths)
            median_val = np.median(lengths)  # Calculate median
            if min_val is not None:
                plot_max_val = max_val if max_val > min_val else min_val + 0.1
                fig.add_trace(go.Bar(
                    x=[plot_max_val - min_val],
                    y=[y - y_offset],
                    base=min_val,
                    name="Human",
                    marker=dict(color=SOFT_ORANGE),
                    orientation='h',
                    width=0.35,
                    hovertemplate=f"<b>Min:</b> {min_val}<br>"
                                f"<b>Median:</b> {median_val}<br>"
                                f"<b>Max:</b> {max_val}<extra></extra>",
                    showlegend=bool(y == 0)
                ))


    # Plot Shortest paths
    for y, lengths in zip(y_positions, aligned_shortest_path_len):
        if lengths:
            fig.add_trace(go.Scatter(
                x=lengths,
                y=[y] * len(lengths),
                mode='markers',
                name="Shortest",
                marker=dict(color=LIGHT_GREEN, size=10, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>Shortest Path Length:</b> {lengths[0]}<extra></extra>",
                showlegend=bool(y == 0)
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="Path Length Comparison",
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
            ticktext=[f"{pair[0]} → {pair[1]}" for pair in selected_ori_dest],
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
    # pio.write_html(fig, "path_length_comparison.html", full_html=True)


def plot_accumulated_distribution_of_ranks_conditioned_on_low_similarity(ai_scores_accumulated, human_scores_accumulated, optimal_scores_accumulated):
    # Plot the accumulated distribution (only first 50 ranks)
    fig = go.Figure()

    # AI path
    fig.add_trace(go.Scatter(
        x=list(range(50)),
        y=ai_scores_accumulated[:50],
        mode='lines',
        name='AI path',
        line=dict(color="rgba(58, 85, 209, 0.7)", width=3)  # PRIMARY_BLUE
    ))

    # Human path
    fig.add_trace(go.Scatter(
        x=list(range(50)),
        y=human_scores_accumulated[:50],
        mode='lines',
        name='Human path',
        line=dict(color="rgba(242, 153, 74, 0.7)", width=3)  # SOFT_ORANGE
    ))

    # Shortest path
    fig.add_trace(go.Scatter(
        x=list(range(50)),
        y=optimal_scores_accumulated[:50],
        mode='lines',
        name='Shortest path',
        line=dict(color="rgba(111, 207, 151, 0.8)", width=3)  # LIGHT_GREEN
    ))

    # Update layout for aesthetics
    fig.update_layout(
        title="Accumulated Distribution of Rank of Neighbors' Degrees Conditioned on Low Similarity with Destination",
        xaxis_title="Rank of neighbors' degrees conditioned on low similarity with destination",
        yaxis_title="Accumulated frequency",
        plot_bgcolor="rgba(240, 244, 255, 1)",  # Background
        paper_bgcolor="rgba(240, 244, 255, 1)",  # Paper background
        font=dict(size=12),
        width=1000,   # Fixed width
        height=800   # Fixed height
    )

    # Update grid line styles
    fig.update_xaxes(showgrid=True, gridcolor="rgba(224, 232, 240, 1)", dtick=5)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(224, 232, 240, 1)")

    # Show the interactive plot
    fig.show()
    # pio.write_html(fig, "pages/accumulated_distribution_of_rank_of_neighbors_conditioned_on_low_similiarity.html", full_html=True)


def plot_mean_degrees_and_closeness(degree_means, score_means):

    # Define fixed colors with transparency
    PRIMARY_BLUE = "rgba(58, 85, 209, 0.7)"  # AI paths - Light Blue with opacity
    SOFT_ORANGE = "rgba(242, 153, 74, 0.7)"  # Human paths - Soft Orange with opacity
    LIGHT_GREEN = "rgba(111, 207, 151, 0.8)" # Shortest paths - Light Green with slight opacity


    # Labels for the groups
    degree_labels = ["Good AI", "Bad AI", "Good Human", "Bad Human", "Optimal"]
    score_labels = ["Good AI", "Bad AI", "Good Human", "Bad Human", "Optimal"]

    # Define colors
    bar_colors = [PRIMARY_BLUE, PRIMARY_BLUE, SOFT_ORANGE, SOFT_ORANGE, LIGHT_GREEN]

    # Plotting Degrees
    fig_degrees = go.Figure()

    fig_degrees.add_trace(go.Bar(
        x=degree_labels,
        y=degree_means,
        marker_color=bar_colors,
        hovertemplate=["%{x}: %{y:.2f}<extra></extra>" for x, y in zip(degree_labels, degree_means)]  # Hover template
    ))

    fig_degrees.update_layout(
        title="Mean Out-Degrees Comparison",
        # xaxis_title="Groups",
        yaxis_title="First Step Out-Degrees",
        plot_bgcolor="rgba(240, 244, 255, 1)",  # Background
        paper_bgcolor="rgba(240, 244, 255, 1)",  # Paper background
        font=dict(size=12),
        width=700,
        height=500
    )

    # Plotting Scores
    fig_scores = go.Figure()

    fig_scores.add_trace(go.Bar(
        x=score_labels,
        y=score_means,
        marker_color=bar_colors,
        hovertemplate=["%{x}: %{y:.2f}<extra></extra>" for x, y in zip(score_labels, score_means)]  # Hover template
    ))

    fig_scores.update_layout(
        title="Mean Closeness Scores Comparison",
        # xaxis_title="Groups",
        yaxis_title="First Step Closeness Scores",
        plot_bgcolor="rgba(240, 244, 255, 1)",  # Background
        paper_bgcolor="rgba(240, 244, 255, 1)",  # Paper background
        font=dict(size=12),
        width=700,
        height=500
    )

    # Show both figures
    fig_degrees.show()
    # pio.write_html(fig_degrees, "pages/mean_out_degrees_comparison.html", full_html=True)

    fig_scores.show()
    # pio.write_html(fig_scores, "pages/mean_closeness_scores_comparison.html", full_html=True)


def plot_updated_task_path_length_comparison(before_tuning_paths, after_4o_tuning_paths, after_4o_mini_tuning_paths, ori_dest, shortest_paths, sample_fraction=0.15, fixed_height_per_pair=50, fixed_width=1400, x_jitter_offset=0.025, seed=58, y_offset=0.15):
    '''
    Plot the path length comparison for before and after improvement paths
    Parameters:
    - before_tuning_paths: dict, the paths before tuning
    - after_4o_tuning_paths: dict, the paths after 4o tuning
    - after_4o_mini_tuning_paths: dict, the paths after 4o mini tuning
    - ori_dest: list, the original and destination pair
    - shortest_paths: dict, the shortest paths
    - sample_fraction: float, the sample fraction
    - fixed_height_per_pair: int, the fixed height per pair
    - fixed_width: int, the fixed width
    - x_jitter_offset: float, the x jitter offset
    - seed: int, the random seed
    - y_offset: float, the y offset

    Returns:
    - None
    '''
    
    # Extract path lengths for each pair
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
    
    # Sort the pairs based on the mean path length before tuning, to better visualize the comparison
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
    # pio.write_html(fig, "pages/path_length_comparison_for_before_and_after_improvement_paths.html", full_html=True)
