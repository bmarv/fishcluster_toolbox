import os
import config
import pandas as pd
import pymysql
import downstream_analyses.cluster_occupancy_dbm as dbm


cluster_occupancy_db = dbm.ClusterOccupancyDBInterface(
    config.DATABASE_NAME,
    config.PROJ_PATH + "/cluster_occupancy/",
    config.PASSWORD,
    config.MYSQL_PATH,
    db_created=True,
)


def query_for_cluster_and_region_f_correlation(cl, reg, sample, parent_dir):
    columns = [
        "row_id",
        "step_size",
        "turning_angle",
        "dist_wall",
        "treatment",
        f"cluster_region_{cl}",
    ]
    data_df = cluster_occupancy_db.select_from_db(
        columns=columns,
        matches={
            f"cluster_region_{cl}": [f"{reg}"],
            "treatment": ["predator", "control"],
        },
        sample=sample,
    )
    data_df.to_csv(f"{parent_dir}/raw_df/cl{cl}_reg{reg}_df.csv")
    data_df_corr = data_df[["step_size", "turning_angle", "dist_wall"]].dropna().corr()
    data_df_corr.to_csv(f"{parent_dir}/corr/cl{cl}_reg{reg}_corr.csv")
    return data_df


def query_transitions(
    output_dir, treatment, experimental_day_start, experimental_day_end
):
    conn = pymysql.connect(
        host="localhost", user="root", password=config.PASSWORD, db=config.DATABASE_NAME
    )
    DB_TABLE_NAME = "cluster_occupancies"
    query = f"""
        SELECT
            row_id,
            cluster_region_5,
            cluster_region_10,
            cluster_region_20
        FROM
            {DB_TABLE_NAME}
        WHERE
            treatment = "{treatment}" AND
            experimental_day BETWEEN {experimental_day_start} AND {experimental_day_end}
        ORDER BY
            row_id;
    """
    print("starting queries")
    with conn.cursor() as cursor:
        cursor.execute(query)
        data = cursor.fetchall()
    conn.close()
    print("queries finished")
    columns = ['row_id', 'cluster_region_5', 'cluster_region_10', 'cluster_region_20']
    df = pd.DataFrame(data, columns=columns)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(
        f"{output_dir}/pe_clusters_all_{treatment}_days_{experimental_day_start}_to{experimental_day_end}_queries.csv"
    )
    print("written out to csv")
    return df
