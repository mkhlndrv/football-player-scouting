# Model Directory

This directory contains the final model and its outputs for finding similar football players based on cluster co-occurrence.

## Files
- `03_main_model.ipynb`: Jupyter notebook with the main model implementation. It includes:
  - **Clustering Process**: Category-specific clustering (e.g., Shooting k=6) under the "Clustering" section.
  - **Similarity Analysis**: Functions to find similar players (e.g., `find_similar_players`) under the "Finding Similar Players" section.
- `co_occurrence_matrix.csv`: A 460x460 matrix where each entry (i, j) represents the number of categories (out of 11) in which players i and j belong to the same cluster.
- `similar_player_pairs.csv`: A table of player pairs with co-occurrence ≥8, including player names, Player_IDs, and co-occurrence counts.

## How to Launch the Model
### Option 1: Rerun the Clustering Process
If you want to go through the clustering process from scratch:

1. **Prepare the Environment**:
   - Ensure Python 3.8+ is installed.
   - Install dependencies from the root `requirements.txt`:
     ```bash
     pip install -r ../requirements.txt
     ```
   - Place the cleaned datasets (e.g., `shots_cleaned.csv`) in `/data/cleaned` as described in `/data/README.md`.

2. **Open the Notebook**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `03_main_model.ipynb`.

3. **Run the Clustering Section**:
   - Navigate to the "Clustering" section in the notebook.
   - Execute the cells to perform category-specific clustering (e.g., for Shooting, Passing).
   - This will generate new cluster labels and save updated datasets.

4. **Proceed to Similarity Analysis**:
   - Continue to the "Finding Similar Players" section to compute the co-occurrence matrix and find similar players.

### Option 2: Find Similar Players Directly
If you just want to find similar players using precomputed outputs:

1. **Prepare the Environment**:
   - Ensure Python 3.8+ is installed.
   - Install dependencies:
     ```bash
     pip install -r ../requirements.txt
     ```
   - Ensure `co_occurrence_matrix.csv` and `similar_player_pairs.csv` are in this directory.

2. **Open the Notebook**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `03_main_model.ipynb`.

3. **Run the Similarity Section**:
   - Navigate to the "Finding Similar Players" section.
   - Execute the cells to load `co_occurrence_matrix.csv` and use the `find_similar_players` function (e.g., `find_similar_players("Kylian Mbappé")`).

4. **Use the Outputs**:
   - Explore `similar_player_pairs.csv` for precomputed pairs with high similarity.
   - Load the matrix with `pandas.read_csv('co_occurrence_matrix.csv', index_col=0)` for custom analysis.

## Dependencies
All required libraries are listed in `../requirements.txt`. The notebook relies on the cleaned datasets in `/data/cleaned` for clustering and the precomputed files (`co_occurrence_matrix.csv`) for similarity analysis.