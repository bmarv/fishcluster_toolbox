-- Drop the existing database if it exists
DROP DATABASE IF EXISTS `%database_name%`;

-- Create the new database for the cluster occupancies
CREATE DATABASE `%database_name%`;
USE `%database_name%`;

-- Create table for cluster occupancies
CREATE TABLE cluster_occupancies (
  row_id INT NOT NULL AUTO_INCREMENT,
  individual VARCHAR(100),
  day VARCHAR(100),
  df_time_index FLOAT,
  positions_x FLOAT,
  positions_y FLOAT,
  step_size FLOAT,
  turning_angle FLOAT,
  dist_wall FLOAT,
  zVals_x FLOAT,
  zVals_y FLOAT,
  cluster_region_5 INT,
  cluster_region_7 INT,
  cluster_region_10 INT,
  cluster_region_20 INT,
  treatment VARCHAR(100),
  experimental_day FLOAT,
  PRIMARY KEY (row_id)
);

