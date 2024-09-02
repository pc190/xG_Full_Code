
# Expected Goals in Football

This repository contains the complete code to run the pre-processing, exploratory data analysis and modelling involved in the research project.

## Structure

- **data/**: Stores the Statsbomb360 event data. Also stores the preprocessed shots_df once run.
- **src/**: Source code for data loading, processing, modeling, and visualization.
  - `data_loader.py`: Module for loading and processing match data.
  - `rathke.py`: Module for Rathke's method and visualization.
  - `fairchild.py`: Module for Fairchild model and analysis.
  - `fcnn.py`: Module for FCNN model.
  - 'main.py': Loads data and runs chosen model.
- **notebooks/**: Jupyter notebook of Exploratory Data Analysis.
- **results/**: Directory for storing xG results of each model.
  
## Usage
1. Set up your environment using `requirements.txt`.
2. Explore the data in `notebooks/`
3. Run the main script in `src/` including the models that you wish to run. Alternatively, explore each model in `src/` individually

