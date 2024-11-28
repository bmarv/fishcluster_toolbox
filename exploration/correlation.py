import os
import time
import exploration.query_helper as query_helper
import exploration.visualization as visualization


def create_directory_structure(cl, date="", sample=10000):
    folders = [
        f"results_cl{cl}_{sample}_{date}/corr",
        f"results_cl{cl}_{sample}_{date}/plots",
        f"results_cl{cl}_{sample}_{date}/raw_df",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return f"results_cl{cl}_{sample}_{date}"


def correlogram_with_hue(cl, date, sample):
    parent_dir = create_directory_structure(cl=cl, date=date, sample=sample)
    for reg in range(1, cl + 1):
        start_time = time.time()
        print(f"running for reg {reg}")
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
            fig_name=f"{parent_dir}/plots/cl{cl}_reg{reg}_raw.pdf",
        )
        visualization.plot_correlogram_with_hue_title_log(
            dataframe=data_df[
                ["step_size", "turning_angle", "dist_wall", "treatment"]
            ].dropna(),
            hue_column="treatment",
            title=f"Cl-Size {cl}, Reg. {reg}",
            log_scale=True,
            fig_name=f"{parent_dir}/plots/cl{cl}_reg{reg}_raw_log_scale.pdf",
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
            fig_name=f"{parent_dir}/plots/cl{cl}_reg{reg}_z-vals.pdf",
        )
        visualization.plot_correlogram_with_hue_title_log(
            dataframe=df_z[
                ["step_size", "turning_angle", "dist_wall", "treatment"]
            ].dropna(),
            hue_column="treatment",
            title=f"Cl-Size {cl}, Reg. {reg}, Z-Vals",
            log_scale=True,
            fig_name=f"{parent_dir}/plots/cl{cl}_reg{reg}" "_z-vals_log_scale.pdf",
        )
        del data_df, temp_df, df_z
        total_time = time.time() - start_time
        print(f"\t => total time needed: {total_time}")
