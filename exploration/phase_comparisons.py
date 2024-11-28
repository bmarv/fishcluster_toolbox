import pandas as pd
import glob
import os
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
