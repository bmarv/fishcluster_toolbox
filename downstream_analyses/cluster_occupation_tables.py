import os
import re
from datetime import datetime
import glob
import hdf5storage
import pandas as pd
from tqdm import tqdm
import wandb

from utils.utils import (
    get_individuals_keys,
    set_parameters,
)


def init_wandb(params):
    wandb.init(
        project="fishcluster_toolbox",
        config=params
    )


def load_projections_for_all_ind(parameters):
    fks = get_individuals_keys(parameters)
    ind_list = []
    for ind in tqdm(fks, total=len(fks), desc='Loading Projections'):
        pfile = glob.glob(parameters.projectPath + f'/Projections/{ind}*_pcaModes.mat')
        pfile.sort()
        days = []
        df_time_index = []
        positions_x = []
        positions_y = []
        step_size = []
        turning_angle = []
        dist_wall = []
        for f in pfile:
            data = hdf5storage.loadmat(f)
            days.append(data['day'])
            df_time_index.append(data['df_time_index'])
            positions_x.append(data['positions'][:, 0])
            positions_y.append(data['positions'][:, 1])
            step_size.append(data['projections'][:, 0])
            turning_angle.append(data['projections'][:, 1])
            dist_wall.append(data['projections'][:, 2])

        ind_dict = {
            'individual': str(ind),
            'day': days,
            'df_time_index': df_time_index,
            'positions_x': positions_x,
            'positions_y': positions_y,
            'step_size': step_size,
            'turning_angle': turning_angle,
            'dist_wall': dist_wall
        }
        ind_list.append(ind_dict)
    return ind_list


def load_wshed_cluster_occ_for_all_ind(parameters):
    wshed_list = []
    cluster_size_list = parameters.kmeans_list
    for cluster_size in tqdm(cluster_size_list, total=len(cluster_size_list), desc='Loading Wshed Clusters'):
        # print(f'cluster size: {cluster_size}')
        wshed_path = f'{parameters.projectPath}/UMAP/zVals_wShed_groups_{cluster_size}.mat'
        wshedfile = hdf5storage.loadmat(wshed_path)
        if cluster_size == cluster_size_list[0]:
            first_cluster_run = True
        else: 
            first_cluster_run = False

        start_index = 0
        for idx, el in enumerate(wshedfile['zValNames'][0]):
            pattern = r'^(.*?)_(\d{8}_\d{6})_.*$'
            match = re.match(pattern, el[0][0])
            individual = match.group(1)
            day = match.group(2)
            n = wshedfile['zValLens'][0][idx]
            # zVals = wshedfile['zValues'][start_index:start_index+n]
            zVals_x = wshedfile['zValues'][start_index:start_index + n][:, 0]
            zVals_y = wshedfile['zValues'][start_index:start_index + n][:, 1]
            cluster_region = wshedfile['watershedRegions'][0][start_index:start_index + n]
            found = False
            for obj in wshed_list:
                if obj['individual'] == individual:
                    found = True
                    if first_cluster_run:
                        obj['day'].append(day)
                        obj['zVals_x'].append(zVals_x)
                        obj['zVals_y'].append(zVals_y)
                    if f'cluster_region_{cluster_size}' in obj:
                        obj[f'cluster_region_{cluster_size}'].append(cluster_region)
                    if f'cluster_region_{cluster_size}' not in obj:
                        obj[f'cluster_region_{cluster_size}'] = [cluster_region]
            if not found:
                wshed_list.append(
                    {
                        'individual': individual,
                        'day': [day],
                        'zVals_x': [zVals_x],
                        'zVals_y': [zVals_y],
                        f'cluster_region_{cluster_size}': [cluster_region]
                    }
                )
            start_index += n
    return wshed_list


def load_meta_data_df(parameters):
    metadata_path = f'{parameters.projectPath}/PE_spreadsheet_Stand_21Sep2023.xlsx'
    meta_data = pd.read_excel(metadata_path, sheet_name='earlyB_spreadsheet')
    meta_data_relevance_filter = meta_data[['date', 'block', 'tank_ID', 'treatment', 'experimental_day']]
    meta_data_relevance_filter['individual'] = meta_data_relevance_filter.apply(lambda row: 'block'+ str(row['block']) + '_' + row['tank_ID'], axis=1)
    meta_data_relevance_filter['day'] = meta_data_relevance_filter.apply(lambda row: '' + str(datetime.strptime(str(row['date']), "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d")) + '_060000', axis=1)
    meta_data_relevance_filter = meta_data_relevance_filter.drop(['block', 'tank_ID', 'date'], axis=1)
    return meta_data_relevance_filter


def merge_data_and_write_to_csv(parameters, ind_list, wshed_list, metadata_df):
    cluster_occupancy_path = f'{parameters.projectPath}/cluster_occupancy'
    if not os.path.exists(cluster_occupancy_path):
        os.makedirs(cluster_occupancy_path)
    for idx, el in tqdm(enumerate(ind_list), total=len(ind_list), desc='Merging and Writing Out Data'):
        if not (ind_list[idx]['individual'] == wshed_list[idx]['individual']):
            print('individuals in raw and embedded lists are not aligned correctly for :')
            print(f"{ind_list[idx]['individual']} == {wshed_list[idx]['individual']} => {ind_list[idx]['individual'] == wshed_list[idx]['individual']}")
            continue
        data = {
            'individual': wshed_list[idx]['individual'],
            'day': [arr[0][0] for arr in ind_list[idx]['day']],
            'df_time_index': [arr[0] for arr in ind_list[idx]['df_time_index']],
            'positions_x': ind_list[idx]['positions_x'],
            'positions_y': ind_list[idx]['positions_y'],
            'step_size': ind_list[idx]['step_size'],
            'turning_angle': ind_list[idx]['turning_angle'],
            'dist_wall': ind_list[idx]['dist_wall'],
            'zVals_x': wshed_list[idx]['zVals_x'],
            'zVals_y': wshed_list[idx]['zVals_y'],
            'cluster_region_5': wshed_list[idx]['cluster_region_5'],
            'cluster_region_7': wshed_list[idx]['cluster_region_7'],
            'cluster_region_10': wshed_list[idx]['cluster_region_10'],
            'cluster_region_20': wshed_list[idx]['cluster_region_20']
        }
        df = pd.DataFrame(data)
        df_combined = df.explode(list(['df_time_index', 'positions_x', 'positions_y', 'step_size', 'turning_angle', 'dist_wall', 'zVals_x', 'zVals_y', 'cluster_region_5', 'cluster_region_7', 'cluster_region_10', 'cluster_region_20']))
        result_df = pd.merge(
            df_combined, 
            metadata_df[metadata_df['individual'] == wshed_list[idx]['individual']],
            on=['individual', 'day'], how='outer'
        )
        result_df.reindex(columns=['individual', 'day', 'experimental_day', 'df_time_index', 'positions_x', 'positions_y', 'step_size', 'turning_angle', 'dist_wall', 'zVals_x', 'zVals_y', 'cluster_region_5', 'cluster_region_7', 'cluster_region_10', 'cluster_region_20', 'treatment'])
        filename = f'{cluster_occupancy_path}/{wshed_list[idx]["individual"]}'
        result_df.to_csv(filename + '.csv', sep=';')
        del result_df, df, data, df_combined


if __name__ == "__main__":
    parameters = set_parameters()
    if parameters.wandb_key:
        init_wandb(parameters)
    ind_list = load_projections_for_all_ind(parameters)
    wshed_list = load_wshed_cluster_occ_for_all_ind(parameters)
    metadata_df = load_meta_data_df(parameters)
    merge_data_and_write_to_csv(
        parameters=parameters,
        ind_list=ind_list,
        wshed_list=wshed_list,
        metadata_df=metadata_df
    )
    if parameters.wandb_key:
        wandb.finish()
