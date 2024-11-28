import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import pearsonr
from pyvis.network import Network
import matplotlib.colors as mcolors


def plot_cluster_counts_f_cluster_size_treatment(
    input_dir, cluster_counts, treatment, cluster_size
):
    plt.figure(figsize=(12, 8))
    for cluster in cluster_counts.columns:
        plt.plot(
            cluster_counts.index,
            cluster_counts[cluster],
            marker="o",
            label=cluster,
        )
    plt.title(
        f"Progression of Cluster Visits ({treatment} Group, Max Cluster Size = {cluster_size})"
    )
    plt.xlabel("Timeframes")
    plt.ylabel("Visit Counts")
    plt.xticks(rotation=45)
    plt.ticklabel_format(style="plain", axis="y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_dir_path = os.path.join(
        input_dir,
        "cluster_visits_p_phase",
    )
    os.makedirs(output_dir_path, exist_ok=True)
    plt.savefig(
        os.path.join(
            output_dir_path,
            f"visits_cluster_size_{cluster_size}_treatment_{treatment}.pdf",
        )
    )


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


def inject_highlight_functionality(html_file, title, pdf_file_path):
    """
    Injects custom JavaScript for highlight/reset functionality into the Pyvis HTML file.
    """
    js_code = """
  <script type="text/javascript">
      var clickedNode = null;

      // Save the original colors when the network is first initialized
      network.on("beforeDrawing", function () {
          network.body.data.nodes.get().forEach(function (node) {
              if (!node.originalColor) {
                  network.body.data.nodes.update({ id: node.id, originalColor: node.color });
              }
          });

          network.body.data.edges.get().forEach(function (edge) {
              if (!edge.originalColor) {
                  network.body.data.edges.update({
                      id: edge.id,
                      originalColor: edge.color,
                      originalFontColor: edge.font ? edge.font.color : "#000000"
                  });
              }
          });
      });

      network.on("click", function (params) {
          if (params.nodes.length > 0) {
              let nodeId = params.nodes[0];
              if (clickedNode === nodeId) {
                  // Reset mode: Restore original colors
                  clickedNode = null;
                  network.body.data.nodes.update(
                      network.body.data.nodes.get().map(node => ({
                          id: node.id,
                          color: node.originalColor
                      }))
                  );
                  network.body.data.edges.update(
                      network.body.data.edges.get().map(edge => ({
                          id: edge.id,
                          color: edge.originalColor,
                          font: { color: edge.originalFontColor }
                      }))
                  );
              } else {
                  // Highlight mode: Dim unrelated nodes and edges
                  clickedNode = nodeId;
                  let connectedEdges = network.getConnectedEdges(nodeId);
                  let connectedNodes = network.getConnectedNodes(nodeId);

                  // Update nodes
                  network.body.data.nodes.update(
                      network.body.data.nodes.get().map(node => ({
                          id: node.id,
                          color: connectedNodes.includes(node.id) || node.id === nodeId
                              ? node.originalColor
                              : "lightgrey"
                      }))
                  );

                  // Update edges
                  network.body.data.edges.update(
                      network.body.data.edges.get().map(edge => ({
                          id: edge.id,
                          color: connectedEdges.includes(edge.id)
                              ? edge.originalColor
                              : "lightgrey",
                          font: { color: connectedEdges.includes(edge.id)
                              ? edge.originalFontColor
                              : "lightgrey" }
                      }))
                  );
              }
          }
      });
  </script>
  """

    # HTML code to embed the PDF
    pdf_embed_code = f"""
    <div style="margin-top: 20px;">
        <iframe 
            src="{pdf_file_path}" 
            style="width: 100%; height: 800px; border: none;">
        </iframe>
    </div>
    """
    heading_code = f"<h1>{title}</h1>"

    # Read the original HTML
    with open(html_file, "r") as file:
        html_content = file.read()

    # Inject the JavaScript before the closing </body> tag
    updated_html = html_content.replace(
        "</body>", heading_code + js_code + pdf_embed_code + "</body>"
    )

    # Save the updated HTML
    with open(html_file, "w") as file:
        file.write(updated_html)


def matrix_to_transition_html(
    transition_matrix,
    title_heading,
    output_html_path,
    pdf_file_path,
    use_percentage=False,
):
    """
    Create an interactive visualization of the transition matrix.

    Args:
        transition_matrix (np.ndarray): Transition matrix to visualize.
        title_heading (str): Title for the interactive visualization.
        output_html_path (str): Path to save the HTML output.
        pdf_file_path (str): Path for embedding a pdf-file with meta info
        use_percentage (bool): If True, display percentages; otherwise, use absolute values.
    """
    # Number of clusters
    num_clusters = transition_matrix.shape[0]

    # # Normalize the matrix for percentages if required
    # if use_percentage:
    #     row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
    #     with np.errstate(
    #         divide="ignore", invalid="ignore"
    #     ):  # Ignore divide-by-zero warnings
    #         transition_matrix = np.nan_to_num(
    #             (transition_matrix.T / row_sums.T).T * 100
    #         )

    net = Network(height="800px", width="100%", notebook=True, directed=True)

    # Adjust physics dynamically for larger networks
    if num_clusters <= 10:
        node_distance = 300
        spring_length = 400
        spring_constant = 0.05
    elif num_clusters <= 20:
        node_distance = 500
        spring_length = 600
        spring_constant = 0.03
    else:
        node_distance = 700
        spring_length = 800
        spring_constant = 0.02

    net.set_options(
        f"""
    {{
      "physics": {{
        "repulsion": {{
          "nodeDistance": {node_distance},
          "springLength": {spring_length},
          "springConstant": {spring_constant}
        }},
        "minVelocity": 0.1,
        "solver": "repulsion"
      }},
      "interaction": {{
        "dragNodes": true,
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 200
      }}
    }}
    """
    )

    color_palette = list(mcolors.TABLEAU_COLORS.values())
    # nodes with proportional scaling and minimum size
    total_visits = np.sum(transition_matrix, axis=0) + np.sum(transition_matrix, axis=1)
    min_size = 15 if num_clusters > 10 else 10
    max_size = 70 if num_clusters > 10 else 50

    if use_percentage:
        scaled_sizes = np.clip(10 + 40 * (total_visits / 100), min_size, max_size)
    else:
        scaled_sizes = np.clip(
            10 + 40 * (np.log1p(total_visits) / np.log1p(np.max(total_visits))),
            min_size,
            max_size,
        )

    node_colors = {
        i + 1: color_palette[i % len(color_palette)] for i in range(num_clusters)
    }

    # isolated nodes
    connected_nodes = set()
    for i in range(num_clusters):
        for j in range(num_clusters):
            if transition_matrix[i, j] > 0:
                connected_nodes.update([i + 1, j + 1])

    # Add only connected nodes
    for i in range(num_clusters):
        if i + 1 in connected_nodes:
            label = f"Cluster {i+1}"
            size = scaled_sizes[i]
            net.add_node(
                i + 1,
                label=label,
                size=size,
                title=f"Cluster {i+1}",
                color=node_colors[i + 1],
            )

    # edges with labels and visible arrows
    for i in range(num_clusters):
        for j in range(num_clusters):
            weight = transition_matrix[i, j]
            if weight > 0:
                label = f"{weight:.1f}%" if use_percentage else f"{weight:.0f}"
                title = f"{label} ({i+1}->{j+1})"
                net.add_edge(
                    i + 1,
                    j + 1,
                    value=weight,
                    title=title,
                    label=label,
                    color=node_colors[j + 1],
                    arrowsize=0.5,
                )

    html_file = f"{output_html_path}_interactive.html"

    net.show(html_file)
    # Inject JavaScript for highlight/reset with light-grey coloring + PDF embedding
    inject_highlight_functionality(html_file, title_heading, pdf_file_path)


def create_overview_html_site(path):
    treatment_list = ["control", "predator"]
    phases = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]
    cluster_size_list = [5, 10, 20]

    html_content = """<!DOCTYPE html>
    <html>
    <head>
        <title>PE - Cluster Transition Analyses</title>
        <style>
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid black; padding: 10px; text-align: center; }
            th { background-color: #f2f2f2; }
            a { text-decoration: none; color: blue; }
        </style>
    </head>
    <body>
        <h1>PE - Cluster Transition Analyses</h1>
    """

    for cluster_size in cluster_size_list:
        html_content += f"    <h2>Cluster Size {cluster_size}</h2>\n"
        html_content += "    <table>\n"
        html_content += "        <thead>\n"
        html_content += "            <tr>\n"
        html_content += "                <th>Treatment</th>\n"

        for start, end in phases:
            html_content += f"                <th>Days {start}-{end}</th>\n"

        html_content += "            </tr>\n"
        html_content += "        </thead>\n"
        html_content += "        <tbody>\n"

        for treatment in treatment_list:
            html_content += "            <tr>\n"
            html_content += f"                <td>{treatment.capitalize()}</td>\n"

            for start, end in phases:
                file_name = f"pe_cluster_{cluster_size}_{treatment}_days_{start}_to{end}_transition_matrix_interactive.html_interactive.html"
                link = f"interactive_html/{file_name}"
                html_content += f'                <td><a href="{link}">View</a></td>\n'

            html_content += "            </tr>\n"

        html_content += "        </tbody>\n"
        html_content += "    </table>\n"

    html_content += "</body>\n</html>"

    output_html_path = os.path.join(path, "overview.html")
    with open(output_html_path, "w") as file:
        file.write(html_content)
