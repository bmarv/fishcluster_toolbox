import pandas as pd
import glob
import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import exploration.visualization as visualization


def cluster_visits_f_cluster_size_treatment(input_dir, cluster_size, treatment):
    cluster_counts = pd.DataFrame()
    file_pattern = os.path.join(
        input_dir, f"pe_cluster_{cluster_size}_{treatment}_days_*_transition_matrix.csv"
    )
    files = glob.glob(file_pattern)

    for file in files:
        file_name = os.path.basename(file)
        timeframe = file_name.split("_days_")[1].split("_")[0]
        df = pd.read_csv(file, index_col=0)
        cluster_sums = df.sum(axis=1)
        cluster_sums.name = f"Days {timeframe}"
        cluster_counts = pd.concat([cluster_counts, cluster_sums], axis=1)
    cluster_counts = cluster_counts.T.sort_index()
    # Re-sort index, converting to integers for proper ordering (if necessary)
    sorted_index = sorted(cluster_counts.index, key=lambda x: int(x.split(" ")[1]))

    cluster_counts = cluster_counts.loc[sorted_index]

    return cluster_counts


def create_cluster_visits_f_all(input_dir):
    cluster_size_list = [5, 10, 20]
    treatment_list = ["control", "predator"]
    for cluster_size in cluster_size_list:
        for treatment in treatment_list:
            cluster_counts = cluster_visits_f_cluster_size_treatment(
                input_dir, cluster_size, treatment
            )
            visualization.plot_cluster_counts_f_cluster_size_treatment(
                input_dir, cluster_counts, treatment, cluster_size
            )


def normalize_matrix_with_smoothing(mat, epsilon=1e-10):
    smoothed = mat + epsilon  # Add small value to avoid division by zero
    return smoothed / smoothed.sum(axis=1, keepdims=True)


# Kullback-Leibler Divergence
def kl_divergence(p, q, epsilon=1e-10):
    p = np.clip(p, epsilon, 1)  # Avoid log(0) issues
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))


# Bures-Wasserstein distance
def bures_wasserstein_distance(p, q, epsilon=1e-10):
    sqrt_p = sqrtm(np.diag(p + epsilon))  # epsilon for stability
    sqrt_q = sqrtm(np.diag(q + epsilon))
    overlap = sqrtm(sqrt_p @ sqrt_q @ sqrt_p)  # non-negative arg for sqrt
    term = np.sum((p - q) ** 2) - 2 * np.trace(overlap)
    term = max(term, 0)
    return np.sqrt(term)


def load_matrices(directory, cluster_size=5):
    matrices = {}
    for file in os.listdir(directory):
        if f"pe_cluster_{cluster_size}_" in file and file.endswith(".csv"):
            parts = file.split("_")
            treatment_group = parts[3]
            phase_days = "Day " + parts[5] + parts[6].split(".")[0].replace("to", "-")
            matrix = pd.read_csv(os.path.join(directory, file), index_col=0).values
            matrices[(phase_days, treatment_group)] = matrix
    return matrices


def extract_days(phase_tuple):
    phase = phase_tuple[0]
    start_day = int(phase.split()[1].split("-")[0])
    return start_day


def plot_divergences_f_cluster_size(directory, cluster_size):
    matrices = load_matrices(directory, cluster_size=cluster_size)
    phases = ["Day 1-7", "Day 8-14", "Day 15-21", "Day 22-28", "Day 29-35", "Day 36-42"]

    fig, axes = plt.subplots(len(phases), 5, figsize=(40, len(phases) * 5))
    cluster_labels = [f"{i + 1}" for i in range(cluster_size)]
    fig.suptitle(f"Cluster Size {cluster_size} - Phase Comparisons", fontsize=40)
    for i, phase in enumerate(phases):
        fig.text(
            0.08,
            1 - (i + 0.5) / len(phases),
            phase,
            ha="center",
            va="center",
            fontsize=20,
            weight="bold",
            rotation=0,
        )
        matrix_control = normalize_matrix_with_smoothing(matrices[(phase, "control")])
        matrix_predator = normalize_matrix_with_smoothing(matrices[(phase, "predator")])

        font_size = 6
        if cluster_size == 20:
            font_size = 3
        visualization.plot_heatmap(
            axes[i, 0], matrix_control, "Control", cluster_labels, font_size=font_size
        )
        visualization.plot_heatmap(
            axes[i, 1], matrix_predator, "Predator", cluster_labels, font_size=font_size
        )
        kl_results = np.zeros((cluster_size, cluster_size))
        js_results = np.zeros((cluster_size, cluster_size))
        bw_results = np.zeros((cluster_size, cluster_size))

        for x in range(cluster_size):
            for y in range(cluster_size):
                kl_results[x, y] = kl_divergence(matrix_control[x], matrix_predator[y])
                js_results[x, y] = jensenshannon(matrix_control[x], matrix_predator[y])
                bw_results[x, y] = bures_wasserstein_distance(
                    matrix_control[x], matrix_predator[y]
                )
        visualization.plot_divergence_heatmap(
            axes[i, 2], kl_results, "KL Divergence", cluster_labels, font_size=font_size
        )
        visualization.plot_divergence_heatmap(
            axes[i, 3], js_results, "JS Divergence", cluster_labels, font_size=font_size
        )
        visualization.plot_divergence_heatmap(
            axes[i, 4], bw_results, "BW Distance", cluster_labels, font_size=font_size
        )

    divergences_dir = os.path.join(directory, "divergences")
    os.makedirs(divergences_dir, exist_ok=True)
    file_name = os.path.join(
        divergences_dir,
        f"PE_Phase_Comparisons_clustersize_{cluster_size}_divergences.pdf",
    )
    plt.savefig(file_name)


def plot_divergences_f_all(directory):
    cluster_size_list = [5, 10, 20]
    for cluster_size in cluster_size_list:
        plot_divergences_f_cluster_size(directory, cluster_size)
