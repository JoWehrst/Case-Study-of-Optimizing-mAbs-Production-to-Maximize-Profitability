
# =======================================================================
# MAIN
# Import data, run model analysis, calculate economics, and optimize
# =======================================================================

import os
import numpy as np
import pandas as pd
import sys

try:
    sys.stdin.reconfigure(encoding="mbcs", errors="replace")
except (AttributeError, ValueError):
    pass

from src.classes import (StandardScalerVAR, ModelAnalysis, EconomicAnalysis)

# Fix Windows console input encoding
sys.stdin.reconfigure(encoding="mbcs", errors="replace")

# =======================================================================
# SETTINGS
# =======================================================================

dataset_folder = "datasets"
figure_folder = "figures"
table_folder = "tables"

os.makedirs(figure_folder, exist_ok=True)
os.makedirs(table_folder, exist_ok=True)

# Selected batches ,
select_batches = [
    169, 193, 196, 223, 235, 262,
    319, 346, 358, 370, 385, 412,
    469, 496, 508, 532, 535, 547,
    619, 631, 658, 685, 694, 697,
    766, 769, 781, 808, 820, 847
]

# Variables included in the multivariable NVAR model
features = [
    "Glucose [mM]",
    "Glutamine [mM]",
    "Lactate [mM]",
    "Ammonia [mM]",
    "VCD [cells/mL]",
    "TCD [cells/mL]",
    "mAb [mg/L]"
]

time_column = "Time [h]"
batch_column = "Batch ID"

# Model-analysis settings
p_max = 5
poly_degree_max = 4
knn_k_max = 16
cv_splits = 3
n_simulations = 100

prediction_interval = 0.95


# =======================================================================
# DATA IMPORT
# =======================================================================

df_ic = pd.read_csv(os.path.join(dataset_folder, "initial_conditions.csv"))

batch_data = []
loaded_batches = []

for batch_id in select_batches:
    file_path = os.path.join(
        dataset_folder, f"batch_{batch_id:04d}.csv"
    )

    if os.path.exists(file_path):
        df_batch = pd.read_csv(file_path)
        df_batch[batch_column] = batch_id

        batch_data.append(df_batch)
        loaded_batches.append(batch_id)

        print(f"Loaded batch {batch_id}: {file_path}")

    else:
        print(f"Missing batch file: {file_path}")

# Check that all requested batches were imported
if len(loaded_batches) != len(select_batches):
    raise FileNotFoundError(
        "One or more selected batch files are missing."
    )

if not batch_data:
    raise FileNotFoundError("No batch data were imported.")

df = pd.concat(batch_data, ignore_index=True)
batch_ids = np.array(loaded_batches)

# Check that the batch data and initial-condition file have the required columns
required_columns = features + [time_column, batch_column]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in batch data: {missing_columns}")

required_ic = [batch_column, "Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
missing_ic = [column for column in required_ic if column not in df_ic.columns]
if missing_ic:
    raise ValueError(f"Missing columns in initial_conditions.csv: {missing_ic}")

df_ic[batch_column] = df_ic[batch_column].astype(str).str.strip()
df_ic = df_ic[df_ic[batch_column].isin([str(b) for b in batch_ids])].copy()
if df_ic[batch_column].duplicated().any():
    raise ValueError("initial_conditions.csv contains duplicate batch IDs.")
missing_ids = [str(b) for b in batch_ids if str(b) not in df_ic[batch_column].values]
if missing_ids:
    raise ValueError(f"Initial conditions missing for Batch IDs: {missing_ids}")

print("\nData import complete.")
print(f"Number of batches: {len(batch_ids)}")
print(f"Number of time-series observations: {len(df)}")
print(f"Process variables: {len(features)}")


# =======================================================================
# MODEL ANALYSIS
# =======================================================================

model_analysis = ModelAnalysis(
    df=df,
    batch_ids=batch_ids,
    features=features,
    time_column=time_column,
    batch_column=batch_column,
    scaler_class=StandardScalerVAR,
    df_ic=df_ic
)

model_results = model_analysis.fit_nvar_models(
    p_max=p_max,
    poly_degree_max=poly_degree_max,
    knn_k_max=knn_k_max,
    cv_splits=cv_splits,
    n_simulations=n_simulations,
    prediction_interval=prediction_interval,
    figure_folder=figure_folder,
    table_folder=table_folder
)

white_noise_results = model_analysis.white_noise_analysis(
    figure_folder=figure_folder,
    table_folder=table_folder
)

# =======================================================================
# ECONOMIC ANALYSIS
# =======================================================================

economic_analysis = EconomicAnalysis(
    model_analysis=model_analysis,
    figure_folder=figure_folder,
    table_folder=table_folder,
    n_simulations=n_simulations
)

economic_results = economic_analysis.run_dashboard()

print("\nAnalysis complete!")