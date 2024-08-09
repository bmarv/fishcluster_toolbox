import re
import numpy as np
import pandas as pd
import glob
from utils.utils import get_individuals_keys, set_parameters


def load_entropy_data(
    parameters,
    time_range=["daily", "hourly"],
    cluster_sizes=["005", "007", "010", "020", "050"],
    clustering_methods=["kmeans", "umap"],
):
    entropy_sum_dict = {}
    for method in clustering_methods:
        entropy_data_dict = {}
        for time in time_range:
            for size in cluster_sizes:
                if method == "kmeans":
                    file_name = (
                        parameters.projectPath
                        + "/plasticity/cluster_entropy_kmeans/"
                        + f"cluster_entropy_kmeans_{time}_{size}.csv"
                    )
                else:
                    file_name = (
                        parameters.projectPath
                        + "/plasticity/cluster_entropy_wshed/"
                        + f"cluster_entropy_wshed_{time}_{size}.csv"
                    )
                entropy_data_dict[f"{time}_{size}"] = pd.read_csv(file_name)
        entropy_sum_dict[method] = entropy_data_dict
    return entropy_sum_dict


def load_coefficient_of_variation_data(
    parameters,
    time_range=["daily", "hourly"],
    accuracies=["050"],
    principal_components=["distance to wall", "turning angle", "step length"],
):
    cv_data_dict = {}
    for time in time_range:
        for accuracy in accuracies:
            for mode in principal_components:
                file_name = (
                    parameters.projectPath
                    + f"/plasticity/cv/cv_{mode}_{time}_ndf_{accuracy}.csv"
                )
                if mode == "distance to wall":
                    mode_shrtnd = "d2w"
                elif mode == "turning angle":
                    mode_shrtnd = "angle"
                elif mode == "step length":
                    mode_shrtnd = "step"
                cv_data_dict[f"{time}_{accuracy}_{mode_shrtnd}"] = pd.read_csv(
                    file_name
                )
    return cv_data_dict


def build_singular_ids_table(
    input_data_dict,
    fish_keys,
    time_constraint,
    measure="entropy",
    clustering_method=None,
    modes=["005", "007", "010", "020", "050"],
    cov_accuracy="050",
):
    output_id_dict = {}
    for id in fish_keys:
        # cv values
        intermediate_dict = {}
        for mode in modes:
            if (measure == "coefficient of variation") or (measure == "cov"):
                identifier_str = f"{time_constraint}_{cov_accuracy}_{mode}"
                column_identifier_str = f"{mode}"
            elif measure == "entropy":
                identifier_str = f"{time_constraint}_{mode}"
                column_identifier_str = f"{clustering_method}_entropy_{mode}"
            mode_df = input_data_dict[f"{identifier_str}"][[id, "Unnamed: 0"]]
            mode_df = mode_df.rename(
                columns={
                    "Unnamed: 0": "timestep",
                    f"{id}": f"{column_identifier_str}"
                }
            )
            mode_df["id"] = id
            intermediate_dict[f"{mode}"] = mode_df
        output_id_dict[f"{id}"] = intermediate_dict
    return output_id_dict


def merge_cov_and_entropy_dicts_to_one_df(
    cov_dict,
    entropy_clustering_method_dict,
    fish_keys,
    cov_modes=["d2w", "angle", "step"],
    entropy_modes=["005", "007", "010", "020", "050"],
    output_file_name=None,
):

    clustering_method_dict = {}
    for entropy_key, entropy_value in entropy_clustering_method_dict.items():
        cv_dfs_list = []
        entropy_dfs_list = []

        for id in fish_keys:
            # cov values
            cv_dict = cov_dict[id]
            cv_df = pd.merge(
                pd.merge(
                    cv_dict[cov_modes[0]],
                    cv_dict[cov_modes[1]],
                    on=["timestep", "id"]
                ),
                cv_dict[cov_modes[2]],
                on=["timestep", "id"],
            )
            cv_dfs_list.append(cv_df)

            # entropy values
            entropy_specified_dict = entropy_value[id]
            if len(entropy_modes) == 5:
                entropy_df = pd.merge(
                    pd.merge(
                        pd.merge(
                            pd.merge(
                                entropy_specified_dict[entropy_modes[0]],
                                entropy_specified_dict[entropy_modes[1]],
                                on=["timestep", "id"],
                            ),
                            entropy_specified_dict[entropy_modes[2]],
                            on=["timestep", "id"],
                        ),
                        entropy_specified_dict[entropy_modes[3]],
                        on=["timestep", "id"],
                    ),
                    entropy_specified_dict[entropy_modes[4]],
                    on=["timestep", "id"],
                )
            elif len(entropy_modes) == 4:
                entropy_df = pd.merge(
                    pd.merge(
                        pd.merge(
                            entropy_specified_dict[entropy_modes[0]],
                            entropy_specified_dict[entropy_modes[1]],
                            on=["timestep", "id"],
                        ),
                        entropy_specified_dict[entropy_modes[2]],
                        on=["timestep", "id"],
                    ),
                    entropy_specified_dict[entropy_modes[3]],
                    on=["timestep", "id"],
                )
            elif len(entropy_modes) == 3:
                entropy_df = pd.merge(
                    pd.merge(
                        entropy_specified_dict[entropy_modes[0]],
                        entropy_specified_dict[entropy_modes[1]],
                        on=["timestep", "id"],
                    ),
                    entropy_specified_dict[entropy_modes[3]],
                    on=["timestep", "id"],
                )
            elif len(entropy_modes) == 2:
                entropy_df = pd.merge(
                    entropy_specified_dict[entropy_modes[0]],
                    entropy_specified_dict[entropy_modes[1]],
                    on=["timestep", "id"],
                )
            entropy_dfs_list.append(entropy_df)

        cv_all_df = pd.concat(cv_dfs_list)
        entropy_all_df = pd.concat(entropy_dfs_list)
        clustering_method_dict.update({str(entropy_key): entropy_all_df})

    clustering_method_dict["kmeans"].set_index(
        ["id", "timestep"], inplace=True
    )
    clustering_method_dict["umap"].set_index(
        ["id", "timestep"], inplace=True
    )
    unified_entropy_dfs = (
        clustering_method_dict["kmeans"]
        .combine_first(clustering_method_dict["umap"])
        .reset_index()
    )

    cols = ["id", "timestep"]
    all_df = pd.merge(cv_all_df, unified_entropy_dfs, on=cols)

    entropy_distinction_list = []
    for entropy_key in entropy_clustering_method_dict.keys():
        entropy_distinction_operation = lambda x: entropy_key + "_entropy_" + x
        entropy_distinction_list += list(
            map(entropy_distinction_operation, entropy_modes)
        )

    all_df_reordered = all_df[
        ["timestep", "id", cov_modes[0], cov_modes[1], cov_modes[2]]
        + entropy_distinction_list
    ]

    if output_file_name is not None:
        all_df_reordered.to_excel(output_file_name)
    return all_df_reordered


def build_and_unify_cov_and_entropy_tables_flow(
    parameters,
    fish_keys,
    time_range_extraction=["daily", "hourly"],
    time_constraint="daily",
    cov_principal_components=[
        "distance to wall", "turning angle", "step length"
    ],
    cov_accuracies=["050"],
    cluster_sizes=["005", "007", "010", "020", "050"],
    clustering_methods=["kmeans", "umap"],
    output_file_name=None,
):

    cov_data_dict = load_coefficient_of_variation_data(
        parameters,
        time_range=time_range_extraction,
        accuracies=cov_accuracies,
        principal_components=cov_principal_components,
    )

    entropy_data_dict = load_entropy_data(
        parameters,
        time_range=time_range_extraction,
        cluster_sizes=cluster_sizes,
        clustering_methods=clustering_methods,
    )

    cov_principal_components = [
        sub.replace("distance to wall", "d2w")
        for sub in cov_principal_components
    ]
    cov_principal_components = [
        sub.replace("turning angle", "angle")
        for sub in cov_principal_components
    ]
    cov_principal_components = [
        sub.replace("step length", "step")
        for sub in cov_principal_components
    ]

    cov_id_features_dict = build_singular_ids_table(
        cov_data_dict,
        fish_keys,
        time_constraint,
        measure="cov",
        modes=cov_principal_components,
        cov_accuracy=cov_accuracies[0],
    )

    entropy_clustering_method_dict = {}
    for method in clustering_methods:
        entropy_clustering_method_dict[method] = build_singular_ids_table(
            entropy_data_dict[method],
            fish_keys,
            time_constraint,
            measure="entropy",
            clustering_method=method,
            modes=cluster_sizes,
        )

    output_df = merge_cov_and_entropy_dicts_to_one_df(
        cov_id_features_dict,
        entropy_clustering_method_dict,
        fish_keys,
        cov_modes=cov_principal_components,
        entropy_modes=cluster_sizes,
        output_file_name=output_file_name,
    )
    return output_df


def extract_features_from_metadata_table(
    date_dict,
    metadata_df,
    time_constraint='daily',
):
    hourly_measure = True if time_constraint == 'hourly' else False
    table_id_dict = {}
    for block_id, block in date_dict.items():
        for compartment_id, compartment in block.items():
            for serial_id, serial in compartment.items():
                table_id = f'block{block_id}_{serial_id}_{compartment_id}'
                serial_id = f'{serial_id}_{compartment_id}'
                id_df = metadata_df[
                    (metadata_df['block'] == int(block_id)) 
                    & (metadata_df['tank_ID'] == serial_id)
                ]
                date_meta_data_list = []
                timestep = 1
                for date_str in serial:
                    for i in range(0, 8 ** hourly_measure):
                        # save nan values as well
                        if date_str == 'nan':
                            out_dict = {
                                'timestep': timestep,
                                'mother_ID': 'nan',
                                'standard_length_cm_beginning_of_week': 'nan',
                                'tank_compartment': 'nan',
                                'tank_position': 'nan',
                                'tank_system': 'nan',
                            }
                        else:
                            # Format the date in 'DD.MM.YY' format
                            year = date_str[2:4]
                            month = date_str[4:6]
                            day = date_str[6:]
                            converted_date = f"{day}.{month}.{year}"
                            outdf = id_df[
                                (id_df['experimental_date'] == converted_date)
                            ]
                            out_dict = {
                                'timestep': timestep,
                                'mother_ID': outdf['mother_ID'].values[0],
                                'standard_length_cm_beginning_of_week': float(
                                    outdf[
                                        'standard_length_cm_beginning_of_week'
                                    ].values[0]
                                ),
                                'tank_compartment': outdf[
                                    'tank_compartment'
                                ].values[0],
                                'tank_position': outdf[
                                    'tank_position'
                                ].values[0],
                                'tank_system': int(outdf[
                                    'tank_system'
                                ].values[0])
                            }
                        timestep += 1
                        date_meta_data_list.append(out_dict)
                table_id_dict[table_id] = date_meta_data_list
    return table_id_dict


def extract_and_store_relevant_dates(
    projections_data_path: str
) -> dict:
    compartment_pattern = r'(front|back)'
    id_pattern = r'_(\d{8})'
    date_pattern = r'_(\d{8})_\d{6}_pcaModes(.)*$'

    date_dict = {}
    for block_nr in ['1', '2']:
        # get filtered block_nr list
        block_els = sorted(
            glob.glob(f'{projections_data_path}/Projections/block{block_nr}*')
        )
        compartment_dict = {}
        for compartment in ['front', 'back']:
            compartment_list = []
            for block_el in block_els:
                # get filtered compartment list
                if ((re.search(compartment_pattern, block_el)).group() == compartment):
                    compartment_list.append(block_el)
            id_list = []
            for comp_el in compartment_list:
                curr_id = (re.search(id_pattern, comp_el)).group(1)
                # get all individual names
                if curr_id not in id_list:
                    id_list.append(curr_id)
                
            curr_id_dict = {}
            for curr_ind in id_list:
                # get filtered individual list
                individual_list = []
                for comp_el2 in compartment_list:
                    if (re.search(id_pattern, comp_el2)).group(1) == curr_ind:
                        individual_list.append(comp_el2)
                date_list = []
                for date_el_p_ind in individual_list:
                    curr_date = (
                        re.search(date_pattern, date_el_p_ind)
                    ).group(1)
                    if curr_date not in date_list:
                        date_list.append(curr_date)

                curr_id_dict[curr_ind] = date_list
            compartment_dict[compartment] = curr_id_dict
        date_dict[block_nr] = compartment_dict
    return date_dict


def reorder_entropy_and_rest_data_by_key(
    input_df,
    key,
    cluster_sizes=['005', '007', '010', '020', '050'],
    clustering_methods=['kmeans', 'umap'],
):
    if (key not in (input_df['id'].values).tolist()):
        return None
    current_id = input_df[input_df['id'] == key]
    entropy_distinction_list = []
    for method in clustering_methods:
        entropy_distinction_operation = lambda x: method + '_entropy_' + x
        entropy_distinction_list += list(
            map(entropy_distinction_operation, cluster_sizes)
        )
    current_id_entropy_df = current_id[entropy_distinction_list]
    current_id_rest = current_id[[
        'timestep', 'id', 'd2w', 'angle', 'step', 'mother_ID',
        'standard_length_cm_beginning_of_week', 'tank_compartment',
        'tank_position', 'tank_system'
    ]]

    df_list = []
    rest_index = 0
    entropy_index = 0
    for index, row in current_id.iterrows():
        rest_index += 1
        entropy_index += 1
        df1 = pd.DataFrame(current_id_rest.iloc[rest_index - 1]).T
        if np.isnan(row['d2w']):
            df2 = pd.DataFrame(current_id_entropy_df.iloc[entropy_index - 1]).T
            df2[:] = np.nan
            entropy_index -= 1
        else:
            df2 = pd.DataFrame(current_id_entropy_df.iloc[entropy_index - 1]).T
        df1['tmp'] = rest_index
        df2['tmp'] = rest_index
        df_out = pd.merge(df1, df2, on='tmp')
        df_out = df_out.drop('tmp', axis=1)
        # establish correct ordering 
        df_out = df_out[
            ['timestep', 'id', 'd2w', 'angle', 'step'] +
            entropy_distinction_list + 
            [
                'mother_ID', 'standard_length_cm_beginning_of_week',
                'tank_compartment', 'tank_position', 'tank_system'
            ]
        ]
        df_list.append(df_out)
    return pd.concat(df_list, axis=0, ignore_index=True)


def unifiy_table_timesteps(
    input_df,
    table_id_dict,
    discard_nan_rows=False,
    cluster_sizes=['005', '007', '010', '020', '050'],
    clustering_methods=['kmeans, umap'],
    output_file_name=None
):
    reordered_df_list = []
    for key in table_id_dict.keys():
        reordered_df_element = reorder_entropy_and_rest_data_by_key(
            input_df,
            key,
            cluster_sizes,
            clustering_methods
        )
        if reordered_df_element is not None:
            reordered_df_list.append(reordered_df_element)
        else:
            print(
                f'Warning: key {key} from metadata not found in training data'
            )

    reordered_df = pd.concat(reordered_df_list, axis=0, ignore_index=True)

    entropy_distinction_list = []
    for method in clustering_methods:
        entropy_distinction_operation = lambda x: method + '_entropy_' + x
        entropy_distinction_list += list(
            map(entropy_distinction_operation, cluster_sizes)
        )
    if discard_nan_rows:
        reordered_df.dropna(
            subset=['d2w', 'angle', 'step'] + entropy_distinction_list,
            how='any',
            inplace=True
        )
    if output_file_name is not None:
        reordered_df.to_excel(output_file_name)
    return reordered_df


def apply_metadata_to_cov_entropy_table_flow(
    cov_entropy_df,
    projections_data_path,
    metadata_df,
    time_constraint,
    output_file_name=None
):
    date_dict = extract_and_store_relevant_dates(
        projections_data_path=projections_data_path
    )
    table_id_dict = extract_features_from_metadata_table(
        date_dict=date_dict,
        metadata_df=metadata_df,
        time_constraint=time_constraint
    )
    result_df = merge_metadata_and_cov_entropy_data(
        cov_entropy_df,
        table_id_dict,
        output_file_name=output_file_name
    )
    return result_df, table_id_dict


def merge_metadata_and_cov_entropy_data(
    cov_entropy_df,
    table_id_dict,
    output_file_name=None
):
    df_list = []
    for id, content in table_id_dict.items():
        # merging the tables element-wise f individuals: cov-entropy+metadata
        df_list.append(
            pd.merge(
                cov_entropy_df[cov_entropy_df['id'] == id],
                pd.DataFrame(table_id_dict[id]),
                on='timestep',
                how='outer'
            )
        )
    # concatenating the tables for every individual
    result_df = pd.concat(df_list, ignore_index=True)
    result_df.replace('nan', np.nan, inplace=True)
    if output_file_name is not None:
        result_df.to_excel(output_file_name)
    return result_df


def unified_table_flow(
    parameters,
    fish_keys,
    time_range_extraction=["daily", "hourly"],
    time_constraint="daily",
    cov_principal_components=[
        "distance to wall", "turning angle", "step length"
    ],
    cov_accuracies=["050"],
    cluster_sizes=["005", "007", "010", "020", "050"],
    clustering_methods=["kmeans", "umap"],
    projections_data_path="/mnt/Projections",
    metadata_path="/mnt/FE_Metadata_for_Entropy_models.xlsx",
    discard_nan_rows=False,
    output_file_name=None,
):
    cov_entropy_df = build_and_unify_cov_and_entropy_tables_flow(
        parameters=parameters,
        fish_keys=fish_keys,
        time_range_extraction=time_range_extraction,
        time_constraint=time_constraint,
        cov_principal_components=cov_principal_components,
        cov_accuracies=cov_accuracies,
        cluster_sizes=cluster_sizes,
        clustering_methods=clustering_methods,
        output_file_name=None,
    )
    metadata_df = pd.read_excel(metadata_path)
    merged_df, table_id_dict = apply_metadata_to_cov_entropy_table_flow(
        cov_entropy_df,
        projections_data_path,
        metadata_df,
        time_constraint,
        output_file_name=None,
    )
    unified_df = unifiy_table_timesteps(
        merged_df,
        table_id_dict,
        discard_nan_rows=discard_nan_rows,
        cluster_sizes=cluster_sizes,
        clustering_methods=clustering_methods,
        output_file_name=output_file_name,
    )

    return unified_df


if __name__ == "__main__":
    parameters = set_parameters()
    fks = get_individuals_keys(parameters)
    parameters = set_parameters()
    projections_data_path = parameters.projectPath
    cluster_sizes_list = parameters.kmeans_list
    cluster_sizes_str_list = [str(num).zfill(3) for num in cluster_sizes_list]
    metadata_path = parameters.projectPath + \
        '/FE_Metadata_for_Entropy_models.xlsx'

    for time_str in ['daily', 'hourly']:
        output_file_name = parameters.projectPath + \
            f'/cpu_{time_str}_table.xlsx'
        print(f'creating CoV & entropy tables for {output_file_name}')
        unified_hourly_df = unified_table_flow(
            parameters=parameters,
            fish_keys=fks,
            time_constraint=time_str,
            cluster_sizes=cluster_sizes_str_list,
            clustering_methods=['kmeans', 'umap'],
            projections_data_path=projections_data_path,
            metadata_path=metadata_path,
            discard_nan_rows=True,
            output_file_name=output_file_name
        )
