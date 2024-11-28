import config
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
