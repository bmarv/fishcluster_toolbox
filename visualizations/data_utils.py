import os
import glob
import pandas as pd
from tqdm import tqdm


def load_data_for_downstream_analyses(csv_dir_path: str) -> pd.DataFrame:
    '''
    Function reads csv-files for every individual, which contains data for the
    whole tracking process.

    WARNING: Loading the data utilizes a lot of memory.

    Args:
        csv_dir_path (str): The directory path where the data is stored.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    '''
    csv_files = glob.glob(
        os.path.join(csv_dir_path, '*.csv')
    )
    df_list = []
    for file in tqdm(csv_files, total=len(csv_files)):
        filename = file.split('/')[-1].split('.')[0]
        df = pd.read_csv(file, sep=';', index_col=[0])
        df.insert(
            loc=0,
            column='fish_id',
            value=filename
        )
        df_list.append(df)
    data = pd.concat(df_list)
    return data
