import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import re

import exploration.query_helper as query_helper


def create_transitions(input_file, cluster_size, output_file):
    data = pd.read_csv(input_file)
    transition_matrix = np.zeros((cluster_size, cluster_size), dtype=int)
    previous_cluster = None
    for _, row in data.iterrows():
        current_cluster = row[f"cluster_region_{cluster_size}"]
        if current_cluster == 0:
            continue
        if pd.isna(row[f"cluster_region_{cluster_size}"]):
            continue
        if previous_cluster is not None and previous_cluster != 0:
            # Update the transition matrix
            transition_matrix[int(previous_cluster) - 1, int(current_cluster) - 1] += 1
        previous_cluster = current_cluster

    transition_matrix_df = pd.DataFrame(
        transition_matrix,
        index=[f"Cluster {i+1}" for i in range(cluster_size)],
        columns=[f"Cluster {i+1}" for i in range(cluster_size)],
    )
    transition_matrix_df.to_csv(output_file, index=True)
    del data, transition_matrix, transition_matrix_df


def queries_for_transition_matrices_f_all(output_dir):
    cluster_size_list = [5, 10, 20]
    treatment_list = ["control", "predator"]
    experimental_day_tuples = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]
    for cluster_size in cluster_size_list:
        for experimental_day_start, experimental_day_end in experimental_day_tuples:
            for treatment in treatment_list:
                query_helper.query_transitions(
                    output_dir,
                    cluster_size,
                    treatment,
                    experimental_day_start,
                    experimental_day_end,
                )


def process_files_for_transition_matrix(
    input_directory, output_dir_transition_matrices
):
    """
    Process all CSV files in the specified directory and generate interactive HTML visualizations.
    """

    # Regex pattern to extract details from file names
    pattern = r"pe_cluster_(\d+)_(control|predator)_days_(\d+)_to(\d+)_queries\.csv"
    os.makedirs(output_dir_transition_matrices, exist_ok=True)
    for file_name in tqdm(
        os.listdir(input_directory), desc="Processing files for transition matrices"
    ):
        match = re.match(pattern, file_name)
        if match:
            cluster_size = int(match.group(1))
            treatment = match.group(2).capitalize()
            day_start = match.group(3)
            day_end = match.group(4)
            output_file = (
                output_dir_transition_matrices
                + f"/pe_cluster_{cluster_size}_{treatment}_days_{day_start}_to{day_end}_transition_matrix.csv"
            )

            # create transition matrix
            create_transitions(
                os.path.join(input_directory, file_name), cluster_size, output_file
            )
