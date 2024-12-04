import os
import argparse
import logging

from exploration import (
    correlation,
    phase_comparisons,
    transition_matrix,
)


logging.basicConfig(
    level=logging.INFO,  # minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # timestamp
    datefmt="%Y-%m-%d %H:%M:%S",  # datetime format
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploration of Trained Data")
    parser.add_argument("--program", required=True, type=str)
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--cluster_size", required=False, type=int)
    parser.add_argument("--subsampling", required=False, type=int)
    args = parser.parse_args()

    logging.info(f"---Running Program: {args.program}---")
    if str(args.program).lower() == "correlation":
        correlation.correlogram_with_hue(
            args.input_dir, args.cluster_size, args.subsampling
        )
    elif str(args.program).lower() == "transition":
        logging.info(
            f"""---Running queries:
                target directory: {args.input_dir}
                subsampling factor: {args.subsampling}
                ---
            """
        )
        transition_matrix.queries_for_transition_matrices_f_all(
            args.input_dir, args.subsampling
        )
        logging.info("---Building transition matrices & Visualizations---")
        transition_matrix.process_files_for_transition_matrix(
            args.input_dir, os.path.join(args.input_dir, "transition_matrix")
        )
        logging.info("---Calculating Divergences---")
        phase_comparisons.plot_divergences_f_all(
            os.path.join(args.input_dir, "transition_matrix")
        )
        logging.info("---Calculating Cluster Visits---")
        phase_comparisons.create_cluster_visits_f_all(
            os.path.join(args.input_dir, "transition_matrix")
        )
