import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, cpu_count

import exploration.query_helper as query_helper
import exploration.visualization as visualization


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
            transition_matrix[int(previous_cluster) - 1, int(current_cluster) - 1] += 1
        previous_cluster = current_cluster

    transition_matrix_df = pd.DataFrame(
        transition_matrix,
        index=[f"Cluster {i+1}" for i in range(cluster_size)],
        columns=[f"Cluster {i+1}" for i in range(cluster_size)],
    )
    transition_matrix_df.to_csv(output_file, index=True)
    del data, transition_matrix, transition_matrix_df


def return_stochastic_matrix_from_transition_matrix(transition_matrix, epsilon=1e-10):
    smoothed = transition_matrix + epsilon  # Add small value to avoid division by zero
    return smoothed * 100 / smoothed.sum(axis=1, keepdims=True)


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


def parallelized_transition_matrix(args):
    file_name, pattern, input_directory, output_dir_transition_matrices = args
    match = re.match(pattern, file_name)
    if match:
        treatment = match.group(1)
        day_start = match.group(2)
        day_end = match.group(3)
        for cluster_size in [5, 10, 20]:
            output_file = (
                output_dir_transition_matrices
                + f"/pe_cluster_{cluster_size}_{treatment}_days_{day_start}_to{day_end}_transition_matrix.csv"
            )
            # create transition matrix
            create_transitions(
                os.path.join(input_directory, file_name), cluster_size, output_file
            )


def process_files_for_transition_matrix(
    input_directory, output_dir_transition_matrices
):
    """
    Process all CSV files in the specified directory and generate interactive HTML visualizations.
    """

    # Regex pattern to extract details from file names
    pattern = r"pe_clusters_all_(control|predator)_days_(\d+)_to(\d+)_queries\.csv"
    os.makedirs(output_dir_transition_matrices, exist_ok=True)
    arguments = [
        (file_name, pattern, input_directory, output_dir_transition_matrices)
        for file_name in os.listdir(input_directory)
    ]
    num_cores = cpu_count() - 1
    with Pool(num_cores) as pool:
        list(
            tqdm(
                pool.imap(
                    parallelized_transition_matrix, arguments
                ),
                desc="Parallelized transition matrices",
                total=len(os.listdir(input_directory))
            )
        )

    pattern_viz = (
        r"pe_cluster_(\d+)_(control|predator)_days_(\d+)_to(\d+)_transition_matrix\.csv"
    )
    output_dir = os.path.join(output_dir_transition_matrices, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(
        os.listdir(output_dir_transition_matrices), desc="Creating Transition PDFs"
    ):
        match = re.match(pattern_viz, file_name)
        if match:
            cluster_size = int(match.group(1))
            treatment = match.group(2).capitalize()
            day_start = match.group(3)
            day_end = match.group(4)

            # Read the transition matrix from the file
            file_path = os.path.join(output_dir_transition_matrices, file_name)
            df = pd.read_csv(file_path, index_col=0)
            transition_matrix = df.values
            output_pdf_path = os.path.join(
                output_dir, os.path.splitext(file_name)[0] + "_visualization.pdf"
            )
            # Create PDF
            with PdfPages(output_pdf_path) as pdf:
                # Absolute values plot
                absolute_title = f"Cluster Size {cluster_size} - {treatment} - Days {day_start} to {day_end} (Absolute Values)"
                visualization.matrix_to_network_pdf(
                    transition_matrix, absolute_title, pdf, use_percentage=False
                )

                # Percentage values plot
                percentage_title = f"Cluster Size {cluster_size} - {treatment} - Days {day_start} to {day_end} (Percentages)"
                transition_matrix = return_stochastic_matrix_from_transition_matrix(
                    transition_matrix
                )
                visualization.matrix_to_network_pdf(
                    transition_matrix, percentage_title, pdf, use_percentage=True
                )

    output_html = os.path.join(output_dir_transition_matrices, "plots")
    os.makedirs(output_html, exist_ok=True)
    for file_name in tqdm(
        os.listdir(output_dir_transition_matrices), desc=" Creating Transition HTMLs"
    ):
        match = re.match(pattern_viz, file_name)
        if match:
            cluster_size = int(match.group(1))
            treatment = match.group(2)
            day_start = match.group(3)
            day_end = match.group(4)

            file_path = os.path.join(output_dir_transition_matrices, file_name)
            df = pd.read_csv(file_path, index_col=0).values
            transition_matrix = return_stochastic_matrix_from_transition_matrix(df)
            output_html_path = os.path.join(
                output_dir_transition_matrices, "interactive_html"
            )
            os.makedirs(output_html_path, exist_ok=True)
            output_html_path = os.path.join(
                output_html_path, os.path.splitext(file_name)[0] + "_interactive.html"
            )

            # PDF file path
            pdf_file_path = (
                f"../plots/{os.path.splitext(file_name)[0]}_visualization.pdf"
            )

            # Generate HTML visualization
            title = f"Cluster Size {cluster_size} - {treatment} - Days {day_start} to {day_end}"
            visualization.matrix_to_transition_html(
                transition_matrix,
                title_heading=title,
                output_html_path=output_html_path,
                pdf_file_path=pdf_file_path,
                use_percentage=True,
            )

    visualization.create_overview_html_site(output_dir_transition_matrices)
