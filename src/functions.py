# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
<<<<<<< HEAD
import os

import sys
import numpy as np
import pandas as pd

def load_initial_conditions(df_ic=None, file_path=None):
    """Return the initial-conditions table, loaded from CSV if needed."""

    if df_ic is not None:
        initial_conditions = df_ic.copy()
    else:
        candidates = []
        if file_path is not None:
            candidates.append(file_path)
        candidates.extend([
            os.path.join("datasets", "initial_conditions.csv"),
            "initial_conditions.csv",
            os.path.join(os.path.dirname(__file__), "..", "datasets", "initial_conditions.csv"),
            os.path.join(os.path.dirname(__file__), "initial_conditions.csv"),
        ])
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(
                "Could not find initial_conditions.csv. Put it in datasets/ or pass file_path."
            )
        initial_conditions = pd.read_csv(path)

    required = ["Batch ID", "Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
    missing = [column for column in required if column not in initial_conditions.columns]
    if missing:
        raise ValueError(f"Missing columns in initial_conditions.csv: {missing}")

    initial_conditions = initial_conditions.copy()
    initial_conditions["Batch ID"] = initial_conditions["Batch ID"].astype(str).str.strip()
    return initial_conditions

def fetch_initial_condition(batch_id, df_ic=None, file_path=None):
    """Return one initial-condition row matching Batch ID (as a pandas Series)."""
    initial_conditions = load_initial_conditions(df_ic, file_path)
    matches = initial_conditions[
        initial_conditions["Batch ID"] == str(batch_id).strip()
    ]
    if matches.empty:
        raise ValueError(f"Batch ID {batch_id} was not found in initial_conditions.csv.")
    return matches.iloc[0]


def progress_bar(c_iter, n_iters, message="", n_updates=100, length_bar=20):
    """Print progress and an optional task description on one line."""
    if n_iters <= 0:
        return
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0 or c_iter >= n_iters:
        progress = min(c_iter / n_iters, 1.0)
        filled = round(length_bar * progress)
        bar = "[" + "=" * filled + " " * (length_bar - filled) + "]"
        sys.stdout.write(f"\r{progress:6.0%} {bar} {message[:100]}")
        sys.stdout.flush()
        if c_iter >= n_iters:
            sys.stdout.write("\n")
            sys.stdout.flush()


def generate_x_lag_cur_ar(p, batch_ids, df, feature, col_batch="Batch ID", col_time="Time [h]"):

    # Store lagged predictors (x_lag) and current values (x_cur) from each batch separately before combining them.
    x_lag = []
    x_cur = []
    t_cur = []

    # Process each batch independently.
    for batch_id in batch_ids:

        # Select rows belonging to the current batch and extract the feature of interest as a NumPy array.
        mask = df.loc[:, col_batch] == batch_id
        data_batch = df.loc[mask, feature].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()

        # Current observations x_(t)
        current_values = data_batch[p:]
        current_times = t_batch[p:]

        # Build autoregressive lag features, e.g., x_(t-1), x_(t-2), ..., x_(t-p).
        lagged_values = []
        for i in range(1, p + 1):
            # Extract the i-th lag, i.e., x_(t-i)
            lagged_values.append(data_batch[p - i:-i])

        # Combine individual lag vectors into a feature matrix where each column represents one lag order.
        lagged_values = np.stack(lagged_values, axis=1)

        # Store this batch's feature matrix and target vector.
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)

    # Concatenate the independently generated batches into overall input (x_lag) and output (x_cur) arrays.
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)

    return x_lag, x_cur, t_cur


def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):

    # Store lagged predictors (x_lag), current values (x_cur), and corresponding times (t_cur)
    # from each batch separately before combining them.
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    # Process each batch independently.
    for batch_id in batch_ids:
        # Select rows belonging to the current batch and extract all variables included in the VAR model.
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        # Current observations x(t).
        # Each row contains the simultaneous values of all variables at a particular time point.
        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        # Build lagged state vectors:
        #
        #     x(t-1), x(t-2), ..., x(t-p)
        #
        # where each lag contains all process variables.
        lagged_values = []
        for i in range(1, p + 1):
            # Extract the i-th lagged state.
            lagged_values.append(x_batch[p - i:-i])

        # Combine all lagged state vectors into a single feature matrix.
        #
        # For example, with two variables and p = 3:
        #
        #     [x1(t-1), x2(t-1), x1(t-2), x2(t-2), x1(t-3), x2(t-3)]
        #
        # Each row therefore contains the lag history used to predict the current state.
        lagged_values = np.concatenate(lagged_values, axis=1)

        # Store this batch's feature matrix, target matrix, and time values.
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    # Concatenate the independently generated batches into overall input and output arrays.
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur

=======

def calculate_average(values):
    return sum(values) / len(values)
>>>>>>> 0bc8a4725e751c81d6c0027649197985d389a77a
