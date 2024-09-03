#!/bin/sh

sed s#%database_name%#"$1"# utils/create_cluster_occupancy_db.sql |
  /usr/local/mysql/bin/mysql --local-infile=1 -u "root" "-p$3"

for file in "$2"*
do
  echo "Processing $file"
  sed s#%database_name%#"$1"# utils/load_cluster_occupancy_file.sql | sed s#%filepath%#"$file"# |
	/usr/local/mysql/bin/mysql --local-infile=1 -u "root" "-p$3"
done