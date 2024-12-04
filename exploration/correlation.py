import os
import time
import logging

import exploration.query_helper as query_helper
import exploration.visualization as visualization


logging.basicConfig(
    level=logging.INFO,  # minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # timestamp
    datefmt="%Y-%m-%d %H:%M:%S",  # datetime format
)


def create_directory_structure(input_dir, cl, sample=10000):
    base_path = os.path.join(input_dir, f"results_cl{cl}_{sample}")
    folders = [
        os.path.join(base_path, "corr"),
        os.path.join(base_path, "plots"),
        os.path.join(base_path, "raw_df"),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return base_path


def correlogram_with_hue(input_dir, cl, sample):
    parent_dir = create_directory_structure(input_dir=input_dir, cl=cl, sample=sample)
    for reg in range(1, cl + 1):
        start_time = time.time()
        logging.info(f"Correlogram running for Region {reg}")
        data_df = query_helper.query_for_cluster_and_region_f_correlation(
            cl=cl, reg=reg, sample=sample, parent_dir=parent_dir
        )
        # original shapes
        visualization.plot_correlogram_with_hue_title_log(
            dataframe=data_df[
                ["step_size", "turning_angle", "dist_wall", "treatment"]
            ].dropna(),
            hue_column="treatment",
            title=f"Cl-Size {cl}, Reg. {reg}",
            log_scale=False,
            fig_name=os.path.join(parent_dir, "plots", f"cl{cl}_reg{reg}_raw.pdf"),
        )
        visualization.plot_correlogram_with_hue_title_log(
            dataframe=data_df[
                ["step_size", "turning_angle", "dist_wall", "treatment"]
            ].dropna(),
            hue_column="treatment",
            title=f"Cl-Size {cl}, Reg. {reg}",
            log_scale=True,
            fig_name=os.path.join(
                parent_dir, "plots", f"cl{cl}_reg{reg}_raw_log_scale.pdf"
            ),
        )
        # z-vals
        temp_df = data_df[
            ["step_size", "turning_angle", "dist_wall", "treatment"]
        ].dropna()
        df_z = (temp_df - temp_df.mean()) / temp_df.std()
        df_z["treatment"] = data_df["treatment"]
        visualization.plot_correlogram_with_hue_title_log(
            dataframe=df_z[
                ["step_size", "turning_angle", "dist_wall", "treatment"]
            ].dropna(),
            hue_column="treatment",
            title=f"Cl-Size {cl}, Reg. {reg}, Z-Vals",
            log_scale=False,
            fig_name=os.path.join(parent_dir, "plots", f"cl{cl}_reg{reg}_z-vals.pdf"),
        )
        visualization.plot_correlogram_with_hue_title_log(
            dataframe=df_z[
                ["step_size", "turning_angle", "dist_wall", "treatment"]
            ].dropna(),
            hue_column="treatment",
            title=f"Cl-Size {cl}, Reg. {reg}, Z-Vals",
            log_scale=True,
            fig_name=os.path.join(
                parent_dir, "plots", f"cl{cl}_reg{reg}_z-vals_log_scale.pdf"
            ),
        )
        del data_df, temp_df, df_z
        total_time = time.time() - start_time
        logging.info(f"\t => total time needed: {total_time}")
