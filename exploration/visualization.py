import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import pearsonr


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def plot_correlogram_with_hue_title_log(
    dataframe, hue_column, title=None, log_scale=False, fig_name=None
):
    palette = sns.color_palette("Set2", len(dataframe[hue_column].unique()))
    hue_colors = dict(zip(dataframe[hue_column].unique(), palette))

    g = sns.PairGrid(dataframe, hue=hue_column, palette="Set2")

    # Diagonal: Density plots
    if log_scale:
        g.map_diag(sns.kdeplot, fill=True, alpha=0.3, log_scale=True)
    else:
        g.map_diag(sns.kdeplot, fill=True, alpha=0.3)

    # Lower: Scatter plots with hue
    g.map_lower(sns.scatterplot, alpha=0.8, s=2.5)

    # Upper: Correlation and significance
    def corrfunc(x, y, **kwargs):
        ax = plt.gca()
        y_pos = 0.85

        # overall correlation
        if not hasattr(ax, "overall_corr_printed"):
            if len(x) >= 2 and len(y) >= 2:
                r_all, p_all = pearsonr(dataframe[x.name], dataframe[y.name])
                ax.text(
                    0.5,
                    y_pos,
                    f"All: {r_all:.5f}{significance_stars(p_all)}",
                    fontsize=10,
                    ha="center",
                    transform=ax.transAxes,
                )
            ax.overall_corr_printed = True

        # per-category correlation
        for category, color in hue_colors.items():
            mask = dataframe[hue_column] == category
            x_cat = x[mask]
            y_cat = y[mask]
            y_pos -= 0.15
            if len(x_cat) >= 2 and len(y_cat) >= 2:
                r_cat, p_cat = pearsonr(x_cat, y_cat)
                ax.text(
                    0.5,
                    y_pos,
                    f"{category}: {r_cat:.5f}{significance_stars(p_cat)}",
                    fontsize=10,
                    ha="center",
                    transform=ax.transAxes,
                    color=color,
                )

    g.map_upper(corrfunc)
    g.add_legend()

    if title:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)

    if log_scale:
        for ax in g.axes.flatten():
            if ax is not None:
                ax.set_xscale("log")
                ax.set_yscale("log")

    if fig_name is not None:
        plt.savefig(fig_name)


def matrix_to_network_pdf(transition_matrix, title, output_pdf, use_percentage=False):
    """
    Visualize the transition matrix as a network graph.
    Save the graph as a page in a PDF.

    Args:
        transition_matrix (np.ndarray): The transition matrix to visualize.
        title (str): The title of the plot.
        output_pdf (PdfPages): PdfPages object to save the plots.
        use_percentage (bool): If True, plot percentages; otherwise, use absolute values.
    """
    G = nx.DiGraph()
    # nodes (clusters)
    num_clusters = transition_matrix.shape[0]
    total_visits = np.sum(transition_matrix, axis=0) + np.sum(transition_matrix, axis=1)

    # Node size scaling
    if use_percentage:
        scaled_sizes = 100 + 600 * (total_visits / 100)
    else:
        # Logarithmic scaling
        scaled_sizes = 200 + 800 * (
            np.log1p(total_visits) / np.log1p(np.max(total_visits))
        )

    for i in range(num_clusters):
        G.add_node(i + 1, size=scaled_sizes[i], label=f"{i + 1}")

    # edges (transitions between clusters)
    for i in range(num_clusters):
        for j in range(num_clusters):
            weight = transition_matrix[i, j]
            if weight > 0:  # Only add edges with transitions
                G.add_edge(i + 1, j + 1, weight=weight)
    # positions for nodes
    pos = nx.circular_layout(G)
    # Adjust plot size dynamically for larger cluster sizes
    if num_clusters > 10:
        plt.figure(figsize=(12, 12))
        font_size = 5
        arrow_size = 10
        alpha = 0.7
    else:
        plt.figure(figsize=(8, 8))
        font_size = 7
        arrow_size = 15
        alpha = 0.9
    cmap = cm.get_cmap("tab10", num_clusters)
    node_colors = [cmap(i) for i in range(num_clusters)]
    node_color_map = {n: node_colors[n - 1] for n in G.nodes}
    # nodes with scaled sizes
    node_sizes = [G.nodes[n]["size"] for n in G.nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=list(node_color_map.values()),
        alpha=alpha,
    )
    # color of edges
    edges = G.edges(data=True)
    drawn_labels = {}
    for u, v, data in edges:
        edge_color = node_color_map[v]
        if u == v:
            # self-loop with increased curvature
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                arrowstyle="->",
                arrowsize=arrow_size,
                edge_color=[edge_color],
                alpha=alpha,
                connectionstyle="arc3,rad=1.0",
            )
            offset_angle = np.pi / 4
            offset_radius = 0.15
            node_pos = pos[u]
            offset_pos = node_pos + offset_radius * np.array(
                [np.cos(offset_angle), np.sin(offset_angle)]
            )
            plt.text(
                offset_pos[0],
                offset_pos[1],
                (
                    f"{data['weight']:.1f}% ({u}->{v})"
                    if use_percentage
                    else f"{data['weight']:.0f} ({u}->{v})"
                ),
                fontsize=font_size,
                color=edge_color,
                ha="center",
                va="center",
            )
        else:
            # regular edge
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                arrowstyle="->",
                arrowsize=arrow_size,
                edge_color=[edge_color],
                alpha=alpha,
                connectionstyle="arc3,rad=0.1",
            )
            label_pos = (pos[u] + pos[v]) / 2
            label_key = tuple(np.round(label_pos, decimals=2))
            if label_key in drawn_labels:
                # label overlap
                label_pos += np.array([0.05, 0.05]) * len(drawn_labels[label_key])
                drawn_labels[label_key].append(data["weight"])
            else:
                drawn_labels[label_key] = [data["weight"]]
            plt.text(
                label_pos[0],
                label_pos[1],
                (
                    f"{data['weight']:.1f}% ({u}->{v})"
                    if use_percentage
                    else f"{data['weight']:.0f} ({u}->{v})"
                ),
                fontsize=font_size,
                color=edge_color,
                ha="center",
                va="center",
            )
    nx.draw_networkx_labels(
        G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=font_size
    )
    plt.title(title, fontsize=font_size + 6)
    plt.axis("off")
    output_pdf.savefig()
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 6) if num_clusters > 10 else (8, 4))
    ax.axis("tight")
    ax.axis("off")
    df = pd.DataFrame(
        transition_matrix,
        index=[f"Cluster {i+1}" for i in range(num_clusters)],
        columns=[f"Cluster {i+1}" for i in range(num_clusters)],
    )
    table_data = df.round(2) if use_percentage else df.astype(int)
    ax.table(
        cellText=table_data.values,
        rowLabels=table_data.index,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="center",
    )
    ax.set_title(
        f"Transition Matrix ({'Percentages' if use_percentage else 'Absolute Values'})",
        fontsize=font_size + 6,
    )
    output_pdf.savefig()
    plt.close()
