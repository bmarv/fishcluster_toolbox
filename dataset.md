# Dataset - Readme

## Overview
This dataset is designed for studying behavioral plasticity in the context of the Amazon clonal molly fish (Poecilia formosa). The dataset captures the dynamic behavior of 45 individual fish over a period of 28 days. The tracking is conducted at a high temporal resolution of 0.2 seconds, focusing on an 8-hour window each day.
The dataset includes time series data for the following key features:
* X-Y Coordinates: The spatial positions of each fish are recorded over time, providing insights into their movement patterns within the experimental environment.
* Step-Length: Information on the distance covered by each fish per time step, offering a quantifiable measure of its activity.
* Turning-Angle: The angles at which the fish change direction, allowing for the analysis of navigation and exploration behaviors.
* Distance-to-the-Wall: The proximity of each fish to the enclosure walls, providing context on habitat preferences and spatial utilization.


## Dataset Description
### Content & Structure
The dataset is organized in MATLAB files, with each file representing the tracking data for a single day of observation for an individual fish. The naming convention for each file follows a structured format:
* Individual Identifier: Unique identifier for each fish in the format (block_tankid_compartment).
* Date: Date of tracking in the format yyyymmdd.
* Tracking Starting Hour: The hour at which tracking for the day commenced, represented as 0600.

The data files include also information detailing the experimental area setup, providing insights into the spatial context of the fish-tanks, the df_time_index and day as well as the fish_key for every datapoint in the following columns:
* area
* day
* df_time_index
* fish_key

Information about the position and the features (=projections) are included in the following columns:
* position
* projections

### Size
The dataset, comprising 1,249 files, totals approximately 3.4 Gigabytes. Each file corresponds to the behavioral tracking data for an individual fish on a specific day, capturing a maximum of 144,000 timesteps. This comprehensive dataset allows for in-depth analyses of behavioral patterns and variations over time.

### Features
####  Time Index
The time index in this dataset begins at 108,000, corresponding to the start of the 6th hour of the day (each hour includes 5 frames * 60 seconds * 60 minutes = 18,000 datapoints). Invalid datapoints, if present, have been excluded from the dataset to prevent potential discrepancies and ensure a continuous time series. Users should be aware that missing timesteps may result from the exclusion of invalid data. 

#### Positions
The dataset includes precise X-Y coordinate tuples representing the positions of each fish over time. These coordinates provide a spatial context for the behavioral observations, allowing researchers to analyze movement patterns and trajectories.

#### Projections
Projections in the dataset are represented as triplets, encompassing the following features:
* Step-Size (cm): This variable quantifies the distance covered by each fish in centimeters per timestep, providing a measure of its activity.
* Turning-Angle (degree): The turning angle indicates the change in direction of the fish's movement over consecutive timesteps, offering insights into navigation and exploration behaviors.
* Distance-to-the-Wall (cm): This feature measures the proximity of each fish to the enclosure walls, offering valuable information on habitat preferences and spatial utilization.

### Data Source
The dataset originates from a carefully designed experimental setting as detailed in the following references: [[1](#1-ehlman-sean-m-et-al-developmental-arcs-of-plasticity-in-whole-movement-repertoires-of-a-clonal-fish-biorxiv-2023-2023-12), [2](#2-u-scherer-s-m-ehlman-d-bierbach-j-krause-m-wolf-reproductive-individuality-of-clonal-fish-raised-in-near-identical-environments-and-its-link-to-early-life-behavioral-individuality-nature-communications-14-7652-2023)]. The study involved the observation of Amazon clonal molly fish (Poecilia formosa) within a controlled environment.
The breeding population consisted of an F0 mother and three F1 offspring, with a total of 45 F2 offspring included in the dataset. The F1 and F2 offspring were produced through mating with Poecilia mexicana male fish.

Observations were conducted using overhead filming within four fish tanks over a period of 28 days. The choice of overhead filming allowed for comprehensive tracking of two dominant dimensions of movement due to relatively shallow water levels.
The study maintained a 12:12-hour light-dark cycle to simulate natural lighting conditions. Fish were fed outside the filming periods to avoid potential disturbances during data collection.
The tracking data was generated using the Biotracker software.

## Data Preprocessing
### Data Cleaning
To enhance the reliability of the dataset, a thorough data cleaning process was applied. Erroneous tracking data, which could potentially introduce inaccuracies, was systematically excluded from the dataset. This careful curation minimizes the impact of outliers and ensures a more accurate representation of the observed behaviors.

### Normalization
Normalization procedures were implemented to standardize the data and account for variations in tank compartments across all individuals. Additionally, a calibration process was conducted to convert pixel values to centimeters, providing a consistent and interpretable metric for spatial coordinates. These normalization steps contribute to the comparability of behavioral metrics across different individuals and experimental conditions.

### Transformation
Originating from the x-y coordinates within each fish-tank, the core features were computed for conducting further analyses:
* Step-Size: The step size, representing the distance covered by each fish per timestep, was computed by analyzing two consecutive datapoints.
* Turning-Angle: The turning angle, indicating changes in the direction of fish movement over three consecutive timesteps, was computed.
* Distance-to-the-Wall: The distance of each fish to the enclosure walls was computed.

## Usage
* loading a single data-file using python:
    ```python
    import hdf5storage
    import pandas as pd
    projection =  hdf5storage.loadmat('/path/to/dataset/<blockid>_<tankid>_<compartmentid>_<date>_060000_pcaModes.mat')
    projection_df = pd.DataFrame([projection])
    # creation of a table-overview for the dataframe
    projection_df_corrected = projection_df.explode(['day', 'fish_key', 'df_time_index']).explode(['df_time_index', 'positions', 'projections'])
    projection_df_corrected['day'] = projection_df_corrected['day'].str[0]
    projection_df_corrected['fish_key'] = projection_df_corrected['fish_key'].str[0]
    projection_df_corrected = projection_df_corrected.reset_index(drop=True)
    ```

---
# References

###### [1] Ehlman, Sean M., et al. "Developmental arcs of plasticity in whole movement repertoires of a clonal fish." bioRxiv (2023): 2023-12.

###### [2] U. Scherer, S. M. Ehlman, D. Bierbach, J. Krause, M. Wolf, Reproductive individuality of clonal fish raised in near-identical environments and its link to early-life behavioral individuality. Nature Communications 14, 7652 (2023). 

