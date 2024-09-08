import glob
from downstream_analyses.plasticity import (
    set_parameters,
    get_individuals_keys,
    for_all_cluster_entropy,
    compute_coefficient_of_variation,
    plasticity_for_file
)
from downstream_analyses.cov_entropy_tables import unified_table_flow
import wandb


def init_wandb(params):
    wandb.init(
        project="fishcluster_toolbox",
        config=params
    )


if __name__ == "__main__":
    parameters = set_parameters()
    if parameters.wandb_key:
        init_wandb(parameters)
    fks = get_individuals_keys(parameters)
    cluster_sizes_list = parameters.kmeans_list
    print('---CALCULATING ENTROPY AND COEFFICIENT OF VARIATION---')
    for_all_cluster_entropy(parameters, fks, cluster_sizes_list)
    for flag in [True, False]:
        compute_coefficient_of_variation(
            parameters,
            fks,
            n_df=10,
            forall=True,
            fit_degree=2,
            by_the_hour=flag
        )
        compute_coefficient_of_variation(
            parameters,
            fks,
            n_df=50,
            forall=True,
            fit_degree=2,
            by_the_hour=flag
        )
    files_daily = sorted(
        glob.glob(f"/{parameters.projectPath}/plasticity/*/*daily*.csv")
    )
    files_hourly = sorted(
        glob.glob(f"/{parameters.projectPath}/plasticity/*/*hourly*.csv")
    )
    for f in files_hourly + files_daily:
        for effect in ["sort", "scatter", ""]:
            fig = plasticity_for_file(f, effect)

    print('---CREATING TABLES FOR COEFFICIENT OF VARIATION AND ENTROPY---')
    cluster_sizes_str_list = [str(num).zfill(3) for num in cluster_sizes_list]
    metadata_path = parameters.projectPath + \
        '/FE_Metadata_for_Entropy_models.xlsx'

    for time_str in ['daily', 'hourly']:
        output_file_name = parameters.projectPath + \
            f'/CoV_Entropy_{time_str}_table.xlsx'
        print(f'creating CoV & entropy tables for {output_file_name}')
        unified_hourly_df = unified_table_flow(
            parameters=parameters,
            fish_keys=fks,
            time_constraint=time_str,
            cluster_sizes=cluster_sizes_str_list,
            clustering_methods=['kmeans', 'umap'],
            projections_data_path=parameters.projectPath,
            metadata_path=metadata_path,
            discard_nan_rows=True,
            output_file_name=output_file_name
        )

    if parameters.wandb_key:
        wandb.finish()
