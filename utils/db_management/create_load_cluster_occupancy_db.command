#!/bin/sh

sed s#%database_name%#"$1"# utils/db_management/create_cluster_occupancy_db.sql |
  $4 --local-infile=1 -u "root" "-p$3"

for file in "$2"*
do
  echo "Processing $file"
  sed s#%database_name%#"$1"# utils/db_management/load_cluster_occupancy_file.sql | sed s#%filepath%#"$file"# |
	$4 --local-infile=1 -u "root" "-p$3"
done