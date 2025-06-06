# Football Player Clustering for Scouting

## Project Overview
This project develops a clustering system to scout football player replacements that maintain tactical consistency. Using player statistics across 11 categories (e.g., Goals, Shooting, Passing), the system clusters players by playing style and identifies similar players based on cluster co-occurrence. The main model outperforms a baseline by providing style-aligned matches (e.g., Kylian Mbappé → Vinícius Jr., Ousmane Dembélé), improving scouting decisions.

### Repository Structure
- `/data`: Datasets (see `data/README.md` for details).
- `/notebooks`: Jupyter notebooks for data analysis, preprocessing, and baseline modeling.
  - `01_data_analysis_and_preprocessing.ipynb`: EDA and data preprocessing.
  - `02_baseline_model.ipynb`: Baseline model (clustering all stats with k=4).
- `/model`: Final model and outputs (see `model/README.md`).
  - `03_main_model.ipynb`: Main model with "Clustering" and "Finding Similar Players" sections.
  - `co_occurrence_matrix.csv`: Co-occurrence matrix.
  - `similar_player_pairs.csv`: Similar player pairs.
- `requirements.txt`: Python dependencies.

## How to Launch
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/football-player-clustering.git
cd football-player-clustering
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Data
Follow the instructions in `/data/README.md` to obtain or generate the datasets (e.g., `shots_cleaned.csv`). Place them in the `/data` directory.

### 4. Run the Notebooks
#### Data Analysis and Baseline
- Open the notebooks in Jupyter Notebook or Google Colab:
  - Start with `notebooks/01_data_analysis_and_preprocessing.ipynb` for EDA and preprocessing.
  - Run `notebooks/02_baseline_model.ipynb` to train and evaluate the baseline model.

#### Main Model and Similarity Analysis
- **Option 1: Rerun Clustering** (if you want to cluster the data again):
  - Ensure cleaned datasets are in `/data/cleaned`.
  - Open `model/03_main_model.ipynb`.
  - Navigate to the "Clustering" section and run the cells to cluster the data.
  - Proceed to the "Finding Similar Players" section to compute similarities.
- **Option 2: Find Similar Players Directly** (using precomputed outputs):
  - Ensure `model/co_occurrence_matrix.csv` and `model/similar_player_pairs.csv` are present.
  - Open `model/03_main_model.ipynb`.
  - Navigate to the "Finding Similar Players" section and run the cells (e.g., `find_similar_players("Kylian Mbappé")`).

To launch Jupyter Notebook:
```bash
jupyter notebook
```

### 5. Use the Model Outputs
- Check `model/co_occurrence_matrix.csv` and `model/similar_player_pairs.csv` for precomputed results.
- Use the `find_similar_players` function in `model/03_main_model.ipynb` for custom queries.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`
- Access to football player stats data (see `/data/README.md`)

## Project Highlights
- **Task Definition**: Scout tactically consistent player replacements.
- **Data Analysis**: EDA on 11 stat categories (e.g., Shooting, Passing).
- **Preprocessing**: Normalized stats per 90 minutes, engineered features (e.g., GCA_Impact).
- **Baseline**: Clustered all stats (k=4), but resulted in mixed styles.
- **Improvement**: Main model clustered by category (e.g., Shooting k=6), achieving style-aligned matches.

## Future Work
- Add age as a feature or filter to refine player matching.
- Incorporate player price data for budget-aware scouting.