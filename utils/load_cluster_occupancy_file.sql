-- Insert data into the table of cluster occupancies
-- filepath is of the form '/Volumes/{path to hard-drive}/cluster_occupancy/block{#}_{########}_{front/back}.csv'
USE `%database_name%`;

LOAD DATA LOCAL INFILE '%filepath%'
INTO TABLE cluster_occupancies
FIELDS TERMINATED BY ';' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@dummy, `individual`, `day`, `df_time_index`, `positions_x`, `positions_y`, `step_size`, `turning_angle`, `dist_wall`, `zVals_x`, `zVals_y`, `cluster_region_5`, `cluster_region_7`, `cluster_region_10`, `cluster_region_20`, `treatment`, `experimental_day`);

SHOW WARNINGS;