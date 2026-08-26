# Data Directory

This directory contains the datasets used for the football player clustering project. Due to privacy and size considerations, raw data sources are not uploaded directly; instead, cleaned and preprocessed files are provided, along with instructions to generate or obtain them.

## Directory Structure
- `/cleaned`: Preprocessed datasets used for EDA and the main model.
- `/original`: Original raw datasets used for initial analysis and the baseline model.
- `player_data_for_clustering.csv`: Merged dataset for the baseline model.

## Datasets
### Cleaned Datasets (/cleaned)
- `xg_stats_cleaned.csv`
- `touches_cleaned.csv`
- `takeons_cleaned.csv`
- `shots_cleaned.csv`
- `shot_creating_actions_cleaned.csv`
- `passing_cleaned.csv`
- `carries_cleaned.csv`
- `attackers_stats_merged_cleaned_2024_2025.csv`
- `aerial_cleaned.csv`
- `goals_cleaned.csv`
- `goal_creating_actions_cleaned.csv`
- `defensive_cleaned.csv`
- `heading.csv`
- `defensive.csv`

Each file includes:
- Player_ID (unique identifier)
- Player (player name)
- Normalized stat columns (e.g., Shots_Total_per90)
- Cluster label columns (e.g., Shooting_Cluster) after running the main model.

### Original Datasets (/original)
- `xg.csv`
- `touches.csv`
- `takeons.csv`
- `shots.csv`
- `shot_creating_actions.csv`
- `passing.csv`
- `carries.csv`
- `attackers_stats_merged_2024_2025.csv`
- `gca.csv`
- `shot_creating_actions.csv`

These are the raw datasets used before cleaning, suitable for initial EDA and baseline model training.

### Baseline Dataset
- `player_data_for_clustering.csv`: Merged dataset with all stats for baseline clustering (k=4).

## How to Obtain the Data
1. **Source**: Original data is sourced from football statistics databases (e.g., FBref, Opta). Download or scrape similar data for the 11 categories listed above.
2. **Preprocessing**: Run `notebooks/01_data_analysis_and_preprocessing.ipynb` to clean the raw data and generate the cleaned CSV files. Place them in `/data/cleaned`.
3. **Baseline Data**: The `player_data_for_clustering.csv` file is generated during preprocessing and saved directly in `/data`.
4. **Clustering**: Run `notebooks/03_main_model.ipynb` to add cluster labels to the cleaned datasets.

## Expected File Sizes
- Cleaned files: ~50-100 KB each.
- Original files: ~40-90 KB each.
- `player_data_for_clustering.csv`: ~143 KB.
- `attackers_stats_merged_cleaned_2024_2025.csv`: ~260 KB.
- `attackers_stats_merged_2024_2025.csv`: ~165 KB.

## Usage
- Use cleaned datasets for the main model and EDA.
- Use original datasets and `player_data_for_clustering.csv` for the baseline model.