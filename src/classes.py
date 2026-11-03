<<<<<<< HEAD

# =======================================================================
# CLASSES
# Model fitting, forecast analysis, economics, and optimization
# =======================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.functions import (
generate_x_lag_cur_var,
    load_initial_conditions,
    fetch_initial_condition,
    progress_bar
)


# =======================================================================
# SCALER
# Apply the same scale to a variable at every lag position.
# =======================================================================

class StandardScalerVAR(BaseEstimator, TransformerMixin):

    def __init__(self, p):
        self.p = p

    def fit(self, x, y=None):
        if self.p < 1:
            raise ValueError("p must be at least 1.")
        if x.shape[1] % self.p != 0:
            raise ValueError("The number of columns must be divisible by p.")

        n_variables = x.shape[1] // self.p
        blocks = [
            x[:, i * n_variables:(i + 1) * n_variables]
            for i in range(self.p)
        ]
        x_combined = np.concatenate(blocks, axis=0)

        mean = np.mean(x_combined, axis=0)
        scale = np.std(x_combined, axis=0)
        variance = np.var(x_combined, axis=0)
        scale[scale == 0] = 1.0

        self.mean_ = np.tile(mean, self.p)
        self.scale_ = np.tile(scale, self.p)
        self.var_ = np.tile(variance, self.p)
        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_


# =======================================================================
# MODEL ANALYSIS
# LOBOCV, hyperparameter selection, recursive forecasts, and residuals.
# =======================================================================

class ModelAnalysis:

    def __init__(
        self, df, batch_ids, features, time_column="Time [h]",
        batch_column="Batch ID", scaler_class=StandardScalerVAR, df_ic=None
    ):
        self.df = df.copy().sort_values([batch_column, time_column])
        self.df_ic = df_ic.copy() if df_ic is not None else None
        self.batch_ids = np.asarray(batch_ids)
        self.features = list(features)
        self.time_column = time_column
        self.batch_column = batch_column
        self.scaler_class = scaler_class
        self.model_results = None
        self.white_noise_results = None
        self.batch_rmse_results = None
        self.best_models = {}

        if self.df_ic is None:
            self.df_ic = load_initial_conditions()

    def _make_model(self, model_type, p):
        if model_type == "Polynomial":
            model = make_pipeline(
                self.scaler_class(p=p),
                PolynomialFeatures(include_bias=False),
                LinearRegression()
            )
            grid = {
                "polynomialfeatures__degree":
                    range(1, self.poly_degree_max + 1)
            }
        else:
            weight = "uniform" if model_type == "KNN uniform" else "distance"
            model = make_pipeline(
                self.scaler_class(p=p),
                KNeighborsRegressor(weights=weight)
            )
            grid = {
                "kneighborsregressor__n_neighbors":
                    range(2, self.knn_k_max + 1)
            }
        return model, grid

    def _fit_best_model(self, model_type, p, batch_ids_train, cv_splits):
        x_lag, x_cur, _, groups = generate_x_lag_cur_var(
            p=p, batch_ids=batch_ids_train, df=self.df,
            features=self.features, col_batch=self.batch_column,
            col_time=self.time_column
        )
        model, grid = self._make_model(model_type, p)
        search = GridSearchCV(
            model, grid, cv=GroupKFold(n_splits=cv_splits),
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(x_lag, x_cur, groups=groups)
        residuals = search.predict(x_lag) - x_cur
        return search, residuals, x_cur

    def _batch_data(self, batch_id):
        data = self.df[
            self.df[self.batch_column] == batch_id
        ].sort_values(self.time_column)
        return (
            data[self.time_column].to_numpy(float),
            data[self.features].to_numpy(float)
        )

    def _initial_condition(self, batch_id):
        row = fetch_initial_condition(batch_id, self.df_ic)
        labels = []

        for name, unit in [
            ("Glucose [mM]", "mM"),
            ("Glutamine [mM]", "mM"),
            ("VCD [cells/mL]", "cells/mL"),
        ]:
            if name in row.index:
                value = float(row[name])
                formatted = (
                    f"{value:.2e}"
                    if name == "VCD [cells/mL]" else f"{value:g}"
                )
                labels.append(f"{name.split(' [')[0]}={formatted} {unit}")

        return row, " | ".join(labels)

    def _forecast(
        self, model, residuals, x_initial, n_steps, n_simulations,
        prediction_interval, p, rng, clip_min, clip_max
    ):
        forecasts = []

        for _ in range(n_simulations):
            states = [x_initial.copy()]

            while len(states) < n_steps:
                lag_states = states[::-1][:p]
                while len(lag_states) < p:
                    lag_states.append(lag_states[-1].copy())

                baseline = model.predict(
                    np.concatenate(lag_states).reshape(1, -1)
                )[0]
                noise = np.array([
                    rng.choice(residuals[:, j])
                    for j in range(residuals.shape[1])
                ])
                states.append(np.clip(
                    baseline + noise, clip_min, clip_max
                ))

            forecasts.append(states)

        forecasts = np.asarray(forecasts)
        alpha = 1 - prediction_interval

        return (
            np.quantile(forecasts, alpha / 2, axis=0),
            np.median(forecasts, axis=0),
            np.quantile(forecasts, 1 - alpha / 2, axis=0)
        )

    def fit_nvar_models(
        self, p_max=5, poly_degree_max=4, knn_k_max=16, cv_splits=3,
        n_simulations=100, prediction_interval=0.95,
        figure_folder="figures", table_folder="tables"
    ):
        os.makedirs(figure_folder, exist_ok=True)
        os.makedirs(table_folder, exist_ok=True)

        self.poly_degree_max = poly_degree_max
        self.knn_k_max = knn_k_max
        model_types = ["Polynomial", "KNN uniform"]
        batch_rmse_records = []

        target_idx = (
            self.features.index("mAb [mg/L]")
            if "mAb [mg/L]" in self.features
            else len(self.features) - 1
        )

        rng = np.random.default_rng(0)
        records = []
        c_iter = 0
        n_iters = len(model_types) * p_max * len(self.batch_ids)

        for model_type in model_types:
            for p in range(1, p_max + 1):

                # Five columns by six rows (30 graphs).
                fig, axes = plt.subplots(
                    5, 6, figsize=(18, 15), dpi=300,
                    squeeze=False, layout="constrained",
                    sharex=True, sharey=True
                )

                rmses = []

                for batch_index, batch_id in enumerate(self.batch_ids):
                    train_ids = np.asarray([
                        b for b in self.batch_ids if b != batch_id
                    ])

                    search, residuals, x_cur_train = self._fit_best_model(
                        model_type, p, train_ids, cv_splits
                    )

                    t, x_test = self._batch_data(batch_id)
                    row, col = divmod(batch_index, 6)
                    ax = axes[row, col]

                    ic_row, ic_label = self._initial_condition(batch_id)
                    x_initial = (
                        x_test[0].copy()
                        if len(x_test)
                        else np.zeros(len(self.features))
                    )

                    for name in [
                        "Glucose [mM]", "Glutamate [mM]",
                        "Glutamine [mM]", "VCD [cells/mL]"
                    ]:
                        if name in self.features and name in ic_row.index:
                            x_initial[self.features.index(name)] = float(
                                ic_row[name]
                            )

                    if len(x_test):
                        lower, median, upper = self._forecast(
                            search, residuals, x_initial, len(t),
                            n_simulations, prediction_interval, p, rng,
                            np.min(x_cur_train, axis=0),
                            np.max(x_cur_train, axis=0)
                        )

                        rmse = root_mean_squared_error(
                            x_test[:, target_idx], median[:, target_idx]
                        )
                        batch_rmse_records.append({
                            "Model": model_type,
                            "p": p,
                            "Batch ID": batch_id,
                            "mAb RMSE [mg/L]": float(rmse)
                        })
                        rmses.append(rmse)

                        ax.scatter(
                            t, x_test[:, target_idx], s=10,
                            alpha=0.7, label="Observed"
                        )
                        ax.plot(
                            t, median[:, target_idx], linewidth=1.5,
                            label="Forecast median"
                        )
                        ax.fill_between(
                            t, lower[:, target_idx], upper[:, target_idx],
                            alpha=0.2,
                            label=f"{prediction_interval:.0%} PI"
                        )
                        rmse_text = f"RMSE: {rmse:.2f} mg/L"
                    else:
                        rmse_text = "RMSE: unavailable"

                    params = search.best_params_

                    if model_type == "Polynomial":
                        parameter_text = (
                            f"Degree={params['polynomialfeatures__degree']}"
                        )
                    else:
                        parameter_text = (
                            f"k={params['kneighborsregressor__n_neighbors']}"
                        )

                    top_text = " | ".join(
                        part for part in [ic_label, parameter_text] if part
                    )

                    ax.text(
                        0.02, 0.97, top_text, transform=ax.transAxes,
                        ha="left", va="top", fontsize=5,
                        bbox={
                            "facecolor": "white", "alpha": 0.8,
                            "edgecolor": "none", "pad": 1.5
                        },
                        clip_on=True
                    )
                    ax.text(
                        0.98, 0.03, rmse_text, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7,
                        bbox={
                            "facecolor": "white", "alpha": 0.8,
                            "edgecolor": "none", "pad": 1.5
                        },
                        clip_on=True
                    )

                    c_iter += 1
                    message = (
                        f"NVAR fitting | Model: {model_type} | p: {p} | "
                        f"Batch: {batch_id} | Parameters: {params}"
                    )
                    progress_bar(c_iter, n_iters, message=message)

                # Hide unused axes if fewer than 30 batches are selected.
                for k in range(len(self.batch_ids), 30):
                    axes.flat[k].set_visible(False)

                for i in range(5):
                    for j in range(6):
                        ax = axes[i, j]
                        if not ax.get_visible():
                            continue

                        if j == 0:
                            ax.set_ylabel("mAb [mg/L]")
                        else:
                            ax.set_ylabel("")

                        if i == 4:
                            ax.set_xlabel("Time [h]")
                        else:
                            ax.set_xlabel("")

                        if (
                            i == 0 and j == 0
                            and ax.get_legend_handles_labels()[0]
                        ):
                            ax.legend(fontsize=6, loc="best")

                fig.suptitle(
                    f"{model_type} NVAR({p}) forecasts by initial conditions"
                )
                fig.savefig(
                    os.path.join(
                        figure_folder,
                        f"forecast_{model_type.lower().replace(' ', '_')}_p{p}.png"
                    ),
                    dpi=300
                )
                plt.close(fig)

                records.append({
                    "Model": model_type,
                    "p": p,
                    "Mean LOBOCV mAb RMSE [mg/L]":
                        float(np.mean(rmses)) if rmses else np.nan,
                    "SD LOBOCV mAb RMSE [mg/L]":
                        float(np.std(rmses)) if rmses else np.nan,
                })

                self.best_models[(model_type, p)] = search.best_params_

        self.model_results = pd.DataFrame(records)
        self.model_results.to_csv(
            os.path.join(table_folder, "nvar_lobocv_results.csv"),
            index=False
        )
        self.batch_rmse_results = pd.DataFrame(batch_rmse_records)

        self.batch_rmse_results.to_csv(
            os.path.join(table_folder, "nvar_lobocv_batch_rmse.csv"),
            index=False
        )
        print(self.model_results.to_string(index=False))
        print(
            "RMSE standard deviation is reported in "
            "'SD LOBOCV mAb RMSE [mg/L]'."
        )
        return self.model_results

    def white_noise_analysis(
        self, figure_folder="figures", table_folder="tables",
        p_max=5, poly_degree_max=4, knn_k_max=16, cv_splits=3
    ):
        """Fit each model and plot residuals, their mean, and a 95% PI."""

        os.makedirs(figure_folder, exist_ok=True)
        os.makedirs(table_folder, exist_ok=True)

        self.poly_degree_max = poly_degree_max
        self.knn_k_max = knn_k_max
        model_types = ["Polynomial", "KNN uniform"]
        target_name = "mAb [mg/L]"

        target_idx = self.features.index(target_name)
        records = []
        c_iter = 0
        n_iters = p_max * len(model_types)

        for p in range(1, p_max + 1):
            fig, axes = plt.subplots(
                1, len(model_types), figsize=(12, 5), dpi=300,
                squeeze=False, layout="constrained", sharey=True
            )

            for j, model_type in enumerate(model_types):
                x_lag, x_cur, times, groups = generate_x_lag_cur_var(
                    p=p, batch_ids=self.batch_ids, df=self.df,
                    features=self.features, col_batch=self.batch_column,
                    col_time=self.time_column
                )

                model, grid = self._make_model(model_type, p)
                search = GridSearchCV(
                    model, grid, cv=GroupKFold(n_splits=cv_splits),
                    scoring="neg_root_mean_squared_error", n_jobs=-1
                )
                search.fit(x_lag, x_cur, groups=groups)
                residuals = (
                    x_cur[:, target_idx]
                    - search.predict(x_lag)[:, target_idx]
                )

                plot_times = np.sort(np.unique(times))
                residual_mean = []
                residual_lower = []
                residual_upper = []

                for time in plot_times:
                    time_residuals = residuals[np.isclose(times, time)]
                    residual_mean.append(np.mean(time_residuals))
                    residual_lower.append(np.percentile(time_residuals, 2.5))
                    residual_upper.append(np.percentile(time_residuals, 97.5))

                ax = axes[0, j]
                ax.scatter(
                    times, residuals, s=18, alpha=0.7, label="Noise",
                    color="tab:orange", edgecolors="black", linewidths=0.4
                )
                ax.plot(
                    plot_times, residual_mean, color="black",
                    linewidth=2, label="Mean"
                )
                ax.fill_between(
                    plot_times, residual_lower, residual_upper,
                    color="tab:blue", alpha=0.3, label="95% PI"
                )
                ax.axhline(0, color="gray", linewidth=0.8)
                ax.set_title(f"{model_type} | p={p}")
                ax.set_xlabel("Time [h]")
                ax.set_ylabel(
                    f"{target_name} (Residuals)" if j == 0 else ""
                )
                ax.legend()

                records.append({
                    "Model": model_type,
                    "p": p,
                    "Best parameters": str(search.best_params_),
                    "Residual mean": float(np.mean(residuals)),
                    "Residual SD": float(np.std(residuals)),
                    "Residual lag-1 correlation": (
                        float(np.corrcoef(
                            residuals[:-1], residuals[1:]
                        )[0, 1])
                        if len(residuals) > 1 else np.nan
                    )
                })

                c_iter += 1
                message = (
                    f"White Noise Analysis | Model: {model_type} | p: {p} | "
                    f"Parameters: {search.best_params_}"
                )
                progress_bar(c_iter, n_iters, message=message)

            fig.suptitle(f"Residual Analysis, NVAR({p})")
            fig.savefig(
                os.path.join(figure_folder, f"residuals_p{p}.png"),
                dpi=300
            )
            plt.close(fig)

        self.white_noise_results = pd.DataFrame(records)
        self.white_noise_results.to_csv(
            os.path.join(table_folder, "white_noise_analysis.csv"),
            index=False
        )
        return self.white_noise_results



# =======================================================================
# ECONOMIC ANALYSIS
# NVAR prediction, economic calculations, and grid search
# =======================================================================

class EconomicAnalysis:

    def __init__(self, model_analysis, figure_folder="figures",
                 table_folder="tables", n_simulations=100):

        self.analysis = model_analysis
        self.figure_folder = figure_folder
        self.table_folder = table_folder
        self.n_simulations = max(1, int(n_simulations))

        os.makedirs(figure_folder, exist_ok=True)
        os.makedirs(table_folder, exist_ok=True)

    # -------------------------------------------------------------------
    # USER INPUT
    # -------------------------------------------------------------------

    @staticmethod
    def _read_integer(prompt, lower, upper):

        while True:
            try:
                value = int(input(prompt))

                if lower <= value <= upper:
                    return value

                print(f"Enter a value between {lower} and {upper}.")

            except ValueError:
                print("Please enter an integer.")

    @staticmethod
    def _read_range(prompt, allowed_min, allowed_max):

        while True:
            try:
                values = input(prompt).strip().split(":")

                if len(values) != 2:
                    raise ValueError

                lower, upper = map(float, values)

                if lower > upper:
                    print("Minimum cannot exceed maximum.")
                    continue

                if lower < allowed_min or upper > allowed_max:
                    print(
                        f"Enter values between {allowed_min:g} "
                        f"and {allowed_max:g}."
                    )
                    continue

                return lower, upper

            except ValueError:
                print("Enter the range in min:max format.")

    def run_dashboard(self):

        print("Intializing Economic Analysis...")

        # Select NVAR model.
        while True:
            choice = input(
                "\nSelect NVAR model (Polynomial/KNN): "
            ).strip().lower()

            if choice in ["poly", "polynomial", "p"]:
                model_type = "Polynomial"
                break

            if choice in ["knn", "knn uniform", "knn-uniform", "k"]:
                model_type = "KNN uniform"
                break

            print("Please enter Polynomial or KNN.")

        p = self._read_integer(
            "Select NVAR lag order p (1-5): ", 1, 5
        )

        # Determine observed initial-condition limits.
        ic = self.analysis.df_ic

        ranges = {
            "Glucose [mM]": (
                ic["Glucose [mM]"].min(),
                ic["Glucose [mM]"].max()
            ),
            "Glutamine [mM]": (
                ic["Glutamine [mM]"].min(),
                ic["Glutamine [mM]"].max()
            ),
            "VCD [cells/mL]": (
                ic["VCD [cells/mL]"].min(),
                ic["VCD [cells/mL]"].max()
            )
        }

        print("\nEnter the initial-condition ranges in min:max format.")

        glucose_bounds = self._read_range(
            f"Glucose [mM] ({ranges['Glucose [mM]'][0]:g}-"
            f"{ranges['Glucose [mM]'][1]:g}): ",
            *ranges["Glucose [mM]"]
        )

        glutamine_bounds = self._read_range(
            f"Glutamine [mM] ({ranges['Glutamine [mM]'][0]:g}-"
            f"{ranges['Glutamine [mM]'][1]:g}): ",
            *ranges["Glutamine [mM]"]
        )

        vcd_bounds = self._read_range(
            f"Initial VCD [cells/mL] ({ranges['VCD [cells/mL]'][0]:g}-"
            f"{ranges['VCD [cells/mL]'][1]:g}): ",
            *ranges["VCD [cells/mL]"]
        )

        return self.analyze_conditions(
            model_type,
            p,
            glucose_bounds,
            glutamine_bounds,
            vcd_bounds
        )

    # -------------------------------------------------------------------
    # MODEL FITTING AND ACCURACY
    # -------------------------------------------------------------------

    def _fit_model(self, model_type, p):

        analysis = self.analysis

        # Use every available batch to train the final model.
        x_lag, x_cur, _, groups = generate_x_lag_cur_var(
            p=p,
            batch_ids=analysis.batch_ids,
            df=analysis.df,
            features=analysis.features,
            col_batch=analysis.batch_column,
            col_time=analysis.time_column
        )

        model, parameter_grid = analysis._make_model(model_type, p)

        n_splits = min(3, len(np.unique(groups)))

        if n_splits < 2:
            raise ValueError("At least two batches are required.")

        search = GridSearchCV(
            model,
            parameter_grid,
            cv=GroupKFold(n_splits=n_splits),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        search.fit(x_lag, x_cur, groups=groups)

        self.model = search.best_estimator_
        self.model_type = model_type
        self.p = p
        self.best_parameters = search.best_params_

        # Prevent negative concentrations and excessive predictions.
        self.upper_caps = np.maximum(
            np.max(x_cur, axis=0) * 5.0,
            1.0
        )

    def _calculate_accuracy(
        self, glucose_bounds, glutamine_bounds, vcd_bounds
    ):

        batch_results = self.analysis.batch_rmse_results.copy()

        if batch_results is None or batch_results.empty:
            raise ValueError(
                "Batch RMSE results are missing. Run "
                "fit_nvar_models() before the economic analysis."
            )

        ic = self.analysis.df_ic.copy()

        # Standardize batch IDs before merging.
        batch_results["Batch ID"] = (
            batch_results["Batch ID"].astype(str).str.strip()
        )

        ic["Batch ID"] = ic["Batch ID"].astype(str).str.strip()

        # Add initial conditions to each validation batch.
        batch_results = batch_results.merge(
            ic[
                [
                    "Batch ID",
                    "Glucose [mM]",
                    "Glutamine [mM]",
                    "VCD [cells/mL]"
                ]
            ],
            on="Batch ID",
            how="left",
            validate="many_to_one"
        )

        # Select the model and lag order used for the economic analysis.
        selected = batch_results[
            (batch_results["Model"] == self.model_type)
            & (batch_results["p"] == self.p)
        ].copy()

        # Only include validation batches inside all three selected ranges.
        selected = selected[
            selected["Glucose [mM]"].between(*glucose_bounds)
            & selected["Glutamine [mM]"].between(*glutamine_bounds)
            & selected["VCD [cells/mL]"].between(*vcd_bounds)
        ].copy()

        if selected.empty:
            raise ValueError(
                "No validation batches fall within all selected ranges. "
                "Expand the ranges and try again."
            )

        accuracy = pd.DataFrame([{
            "Model": self.model_type,
            "NVAR p": self.p,
            "Number of validation batches": len(selected),
            "Mean mAb RMSE [mg/L]":
                selected["mAb RMSE [mg/L]"].mean(),
            "SD mAb RMSE [mg/L]":
                selected["mAb RMSE [mg/L]"].std(ddof=1)
        }])

        return accuracy, selected

    # -------------------------------------------------------------------
    # PREDICTION TIME GRID AND INITIAL CONDITIONS
    # -------------------------------------------------------------------

    def _set_time_grid(self):

        differences = []

        for batch_id in self.analysis.batch_ids:

            batch = self.analysis.df[
                self.analysis.df[self.analysis.batch_column] == batch_id
            ].sort_values(self.analysis.time_column)

            times = batch[self.analysis.time_column].to_numpy(float)
            differences.extend(np.diff(times))

        differences = np.asarray(differences)
        differences = differences[differences > 0]

        if len(differences) == 0:
            raise ValueError("Could not determine the time interval.")

        dt_hours = np.median(differences)
        max_hours = self.analysis.df[self.analysis.time_column].max()

        self.time_days = (
            np.arange(0, max_hours + dt_hours * 0.5, dt_hours) / 24.0
        )

        self.time_days = self.time_days[
            self.time_days <= max_hours / 24.0 + 1e-9
        ]

    def _initial_state(self, glucose, glutamine, vcd):

        analysis = self.analysis

        # Sort batches by time and retrieve the first observation
        # from each batch.
        initial_data = (
            analysis.df
            .sort_values([
                analysis.batch_column,
                analysis.time_column
            ])
            .groupby(analysis.batch_column)
            .first()
        )

        # Calculate the average initial state across all batches.
        # Create a writable NumPy array so values can be modified.
        state = np.array(
            initial_data[analysis.features].mean(),
            dtype=float,
            copy=True
        )

        # Replace the three optimized initial conditions.
        initial_values = {
            "Glucose [mM]": glucose,
            "Glutamine [mM]": glutamine,
            "VCD [cells/mL]": vcd
        }

        # Update the corresponding positions in the initial state.
        for feature, value in initial_values.items():

            if feature in analysis.features:
                feature_index = analysis.features.index(feature)

                state[feature_index] = value

        return state
    # -------------------------------------------------------------------
    # RECURSIVE NVAR PREDICTION
    # -------------------------------------------------------------------

    def _predict_profile(self, glucose, glutamine, vcd):

        initial_state = self._initial_state(
            glucose, glutamine, vcd
        )

        n_steps = len(self.time_days)
        n_features = len(self.analysis.features)

        profile = np.zeros((n_steps, n_features))
        profile[0] = initial_state

        for step in range(1, n_steps):

            # Arrange lagged states as x(t-1), x(t-2), ..., x(t-p).
            lag_states = []

            for lag in range(1, self.p + 1):

                previous_step = step - lag

                if previous_step < 0:
                    lag_states.append(initial_state.copy())
                else:
                    lag_states.append(profile[previous_step].copy())

            lag_vector = np.concatenate(lag_states).reshape(1, -1)

            prediction = self.model.predict(lag_vector)[0]

            # Prevent negative concentrations and numerical blow-up.
            prediction = np.clip(
                prediction,
                0.0,
                self.upper_caps
            )

            profile[step] = prediction

        return profile

    # -------------------------------------------------------------------
    # ECONOMIC CALCULATIONS
    # -------------------------------------------------------------------

    @staticmethod
    def _calculate_economics(glucose, glutamine, vcd, mab, time):

        cost = (
            10.0 * glucose
            + 20.0 * glutamine
            + 0.0005 * vcd
        )

        value = 1000.0 * mab
        profit = value - cost

        stp = profit / (time + 1.0)
        sty = mab / (time + 1.0)

        return cost, value, profit, stp, sty

    def _evaluate_condition(self, glucose, glutamine, vcd):

        profile = self._predict_profile(glucose, glutamine, vcd)

        mab_index = self.analysis.features.index("mAb [mg/L]")
        mab = profile[:, mab_index]

        cost = (
            10.0 * glucose
            + 20.0 * glutamine
            + 0.0005 * vcd
        )

        profit = 1000.0 * mab - cost
        stp = profit / (self.time_days + 1.0)
        sty = mab / (self.time_days + 1.0)

        # Find the two independent optimal harvest times.
        profit_index = np.argmax(profit)
        stp_index = np.argmax(stp)

        cost_stp, value_stp, profit_stp, max_stp, sty_stp = (
            self._calculate_economics(
                glucose,
                glutamine,
                vcd,
                mab[stp_index],
                self.time_days[stp_index]
            )
        )

        _, _, max_profit, _, _ = self._calculate_economics(
            glucose,
            glutamine,
            vcd,
            mab[profit_index],
            self.time_days[profit_index]
        )

        record = {
            "Glucose [mM]": glucose,
            "Glutamine [mM]": glutamine,
            "Initial VCD [cells/mL]": vcd,

            "Harvest time for maximum profit [d]":
                self.time_days[profit_index],
            "mAb at maximum profit [mg/L]":
                mab[profit_index],
            "Maximum profit [$/L]": max_profit,

            "Harvest time for maximum STP [d]":
                self.time_days[stp_index],
            "mAb at maximum STP [mg/L]":
                mab[stp_index],
            "Profit at maximum STP [$/L]": profit_stp,
            "Maximum space-time profit [$/L/day]": max_stp,
            "Space-time yield [mg/L/day]": sty_stp,

            "Cell culture cost [$/L]": cost_stp,
            "Cell culture value at maximum STP [$/L]": value_stp
        }

        economics = {
            "Profit [$/L]": profit,
            "Space-time profit [$/L/day]": stp,
            "Space-time yield [mg/L/day]": sty
        }

        return record, profile, economics

    # -------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------

    def _simulate_economic_uncertainty(self, profile, record, mean_rmse):

        mab_index = self.analysis.features.index("mAb [mg/L]")
        mab = profile[:, mab_index]
        rng = np.random.default_rng(42)
        simulated_mab = np.maximum(
            rng.normal(mab, mean_rmse, size=(self.n_simulations, len(mab))),
            0.0
        )
        glucose = record["Glucose [mM]"]
        glutamine = record["Glutamine [mM]"]
        vcd = record["Initial VCD [cells/mL]"]
        cost = 10.0 * glucose + 20.0 * glutamine + 0.0005 * vcd
        duration = self.time_days + 1.0
        value = 1000.0 * simulated_mab
        profit = value - cost
        stp = profit / duration
        sty = simulated_mab / duration
        q = [2.5, 97.5]
        return {
            "mAb": simulated_mab,
            "value_lower": np.percentile(value, q[0], axis=0),
            "value_upper": np.percentile(value, q[1], axis=0),
            "stp_lower": np.percentile(stp, q[0], axis=0),
            "stp_upper": np.percentile(stp, q[1], axis=0),
            "sty_lower": np.percentile(sty, q[0], axis=0),
            "sty_upper": np.percentile(sty, q[1], axis=0),
            "mab_lower": np.percentile(simulated_mab, q[0], axis=0),
            "mab_upper": np.percentile(simulated_mab, q[1], axis=0),
        }

    def _generate_figures(self, record, profile, economics, mean_rmse,
                          uncertainty):

        mab_index = self.analysis.features.index("mAb [mg/L]")
        mab = profile[:, mab_index]

        harvest_time = record["Harvest time for maximum STP [d]"]
        lower_bound = uncertainty["mab_lower"]
        upper_bound = uncertainty["mab_upper"]

        # ---------------------------------------------------------------
        # mAb production profile with approximate 95% interval.
        # ---------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

        ax.plot(
            self.time_days,
            mab,
            linewidth=2,
            label="Predicted mAb concentration"
        )

        ax.fill_between(
            self.time_days,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label=f"95% prediction interval ({self.n_simulations} simulations)"
        )

        ax.axvline(
            harvest_time,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Optimal harvest: Day {harvest_time:.2f}"
        )

        ax.set_xlabel("Time [days]")
        ax.set_ylabel("mAb concentration [mg/L]")
        ax.set_title("Optimal mAb Production Profile")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.figure_folder,
                "optimal_mab_production_profile.png"
            )
        )

        plt.close(fig)

        # ---------------------------------------------------------------
        # Profit and space-time profit versus time.
        # ---------------------------------------------------------------

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(9, 9),
            dpi=300,
            sharex=True
        )

        ax1.plot(
            self.time_days,
            economics["Profit [$/L]"],
            linewidth=2,
            label="Profit"
        )

        ax1.axvline(
            harvest_time,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Optimal harvest: Day {harvest_time:.2f}"
        )

        ax1.set_ylabel("Profit [$/L]")
        ax1.set_title("Profit versus Time")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(
            self.time_days,
            economics["Space-time profit [$/L/day]"],
            linewidth=2,
            label="Space-time profit"
        )
        ax2.fill_between(
            self.time_days,
            uncertainty["stp_lower"],
            uncertainty["stp_upper"],
            alpha=0.2,
            label="95% prediction interval"
        )

        ax2.axvline(
            harvest_time,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Optimal harvest: Day {harvest_time:.2f}"
        )

        ax2.set_xlabel("Time [days]")
        ax2.set_ylabel("Space-time profit [$/L/day]")
        ax2.set_title("Space-Time Profit versus Time")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.figure_folder,
                "profit_vs_time.png"
            )
        )

        plt.close(fig)

        # ---------------------------------------------------------------
        # Cell culture value and mAb space-time yield with 95% intervals.
        # ---------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 8), dpi=300, sharex=True
        )
        value = 1000.0 * mab
        ax1.plot(self.time_days, value, linewidth=2, label="Cell culture value")
        ax1.fill_between(
            self.time_days, uncertainty["value_lower"],
            uncertainty["value_upper"], alpha=0.2,
            label="95% prediction interval"
        )
        ax1.axvline(harvest_time, color="red", linestyle="--", linewidth=1.5)
        ax1.set_ylabel("Cell culture value [$/L]")
        ax1.set_title("Cell Culture Value and mAb Space-Time Yield")
        ax1.legend()
        ax1.grid(alpha=0.3)

        sty = mab / (self.time_days + 1.0)
        ax2.plot(self.time_days, sty, linewidth=2, label="mAb space-time yield")
        ax2.fill_between(
            self.time_days, uncertainty["sty_lower"],
            uncertainty["sty_upper"], alpha=0.2,
            label="95% prediction interval"
        )
        ax2.axvline(harvest_time, color="red", linestyle="--", linewidth=1.5,
                    label=f"Optimal harvest: Day {harvest_time:.2f}")
        ax2.set_xlabel("Time [days]")
        ax2.set_ylabel("mAb space-time yield [mg/(L·d)]")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, "value_and_space_time_yield.png"))
        plt.close(fig)

    # -------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------

    def _save_results(
            self, ranked_stp, accuracy, validation_batches,
            bounds, best_profile, mean_rmse, best_stp, uncertainty
    ):

        # Save economic results ranked by space-time profit only.
        ranked_stp.to_csv(
            os.path.join(
                self.table_folder,
                "economic_ranked_by_space_time_profit.csv"
            ),
            index=False
        )

        # Model information and selected ranges.
        model_info = pd.DataFrame([{
            "Model": self.model_type,
            "NVAR lag order": self.p,
            "Best parameters": str(self.best_parameters),
            "Glucose range [mM]": str(bounds[0]),
            "Glutamine range [mM]": str(bounds[1]),
            "VCD range [cells/mL]": str(bounds[2]),
            "Glucose increment [mM]": 5.0,
            "Glutamine increment [mM]": 1.0,
            "VCD increment [cells/mL]": 200000.0,
            "Grid combinations": len(ranked_stp)
        }])

        model_info.to_csv(
            os.path.join(
                self.table_folder,
                "economic_model_information.csv"
            ),
            index=False
        )

        # Model accuracy.
        accuracy.to_csv(
            os.path.join(
                self.table_folder,
                "economic_model_accuracy.csv"
            ),
            index=False
        )

        # Validation batches.
        validation_batches.to_csv(
            os.path.join(
                self.table_folder,
                "economic_validation_batches.csv"
            ),
            index=False
        )

        # Predicted profile with 95% prediction interval.
        profile_data = pd.DataFrame(
            best_profile,
            columns=self.analysis.features
        )

        mab_index = self.analysis.features.index("mAb [mg/L]")
        mab = profile_data["mAb [mg/L]"]

        profile_data.insert(0, "Time [days]", self.time_days)

        profile_data["mAb lower 95% interval [mg/L]"] = uncertainty["mab_lower"]
        profile_data["mAb upper 95% interval [mg/L]"] = uncertainty["mab_upper"]
        profile_data["Space-time profit lower 95% interval [$/L/day]"] = uncertainty["stp_lower"]
        profile_data["Space-time profit upper 95% interval [$/L/day]"] = uncertainty["stp_upper"]
        profile_data["mAb space-time yield lower 95% interval [mg/L/day]"] = uncertainty["sty_lower"]
        profile_data["mAb space-time yield upper 95% interval [mg/L/day]"] = uncertainty["sty_upper"]

        harvest_index = int(np.argmin(np.abs(
            self.time_days - best_stp["Harvest time for maximum STP [d]"]
        )))
        summary = pd.DataFrame([{
            "Recommended initial glucose [mM]": best_stp["Glucose [mM]"],
            "Recommended initial glutamine [mM]": best_stp["Glutamine [mM]"],
            "Recommended initial viable cell density [cells/mL]": best_stp["Initial VCD [cells/mL]"],
            "Recommended harvest time [d]": self.time_days[harvest_index],
            "Estimated cell culture cost [$/L]": best_stp["Cell culture cost [$/L]"],
            "Estimated cell culture value at harvest [$/L]": best_stp["Cell culture value at maximum STP [$/L]"],
            "Cell culture value lower 95% PI [$/L]": uncertainty["value_lower"][harvest_index],
            "Cell culture value upper 95% PI [$/L]": uncertainty["value_upper"][harvest_index],
            "Estimated space-time profit [$/L/day]": best_stp["Maximum space-time profit [$/L/day]"],
            "Space-time profit lower 95% PI [$/L/day]": uncertainty["stp_lower"][harvest_index],
            "Space-time profit upper 95% PI [$/L/day]": uncertainty["stp_upper"][harvest_index],
            "Estimated mAb space-time yield [mg/L/day]": best_stp["Space-time yield [mg/L/day]"],
            "mAb space-time yield lower 95% PI [mg/L/day]": uncertainty["sty_lower"][harvest_index],
            "mAb space-time yield upper 95% PI [mg/L/day]": uncertainty["sty_upper"][harvest_index],
            "Number of Simulations": self.n_simulations,
        }])
        summary.to_csv(os.path.join(self.table_folder, "recommended_economic_conditions.csv"), index=False)

        profile_data.to_csv(
            os.path.join(
                self.table_folder,
                "optimal_mab_production_profile.csv"
            ),
            index=False
        )

        print(f"\nEconomic CSV files saved in: {self.table_folder}")

    # -------------------------------------------------------------------
    # MAIN ECONOMIC ANALYSIS
    # -------------------------------------------------------------------

    def analyze_conditions(
        self, model_type, p, glucose_bounds, glutamine_bounds, vcd_bounds
    ):

        print("\nFitting selected NVAR model using all batches...")
        self._fit_model(model_type, p)

        print("Calculating model accuracy within selected ranges...")

        accuracy, validation_batches = self._calculate_accuracy(
            glucose_bounds,
            glutamine_bounds,
            vcd_bounds
        )

        print(
            f"Validation batches included: {len(validation_batches)}"
        )

        print("Preparing prediction time grid...")
        self._set_time_grid()

        # Generate grid-search values.
        glucose_levels = np.arange(
            glucose_bounds[0], glucose_bounds[1] + 1e-9, 5.0
        )
        glutamine_levels = np.arange(
            glutamine_bounds[0], glutamine_bounds[1] + 1e-9, 1.0
        )
        vcd_levels = np.arange(
            vcd_bounds[0], vcd_bounds[1] + 1e-9, 200000.0
        )

        # Include upper limits when they do not fall on the increment.
        if not np.isclose(glucose_levels[-1], glucose_bounds[1]):
            glucose_levels = np.append(glucose_levels, glucose_bounds[1])

        if not np.isclose(glutamine_levels[-1], glutamine_bounds[1]):
            glutamine_levels = np.append(
                glutamine_levels, glutamine_bounds[1]
            )

        if not np.isclose(vcd_levels[-1], vcd_bounds[1]):
            vcd_levels = np.append(vcd_levels, vcd_bounds[1])

        total = (
            len(glucose_levels)
            * len(glutamine_levels)
            * len(vcd_levels)
        )

        print("\nGRID SEARCH")
        print("-" * 50)
        print(f"Glucose values: {glucose_levels}")
        print(f"Glutamine values: {glutamine_levels}")
        print(f"VCD values: {vcd_levels}")
        print(f"Total combinations: {total}")

        records = []
        best_stp_profile = None
        best_stp_economics = None
        best_stp_value = -np.inf

        for i, (glucose, glutamine, vcd) in enumerate(
            itertools.product(
                glucose_levels,
                glutamine_levels,
                vcd_levels
            ),
            start=1
        ):

            record, profile, economics = self._evaluate_condition(
                glucose, glutamine, vcd
            )

            records.append(record)

            stp_value = record["Maximum space-time profit [$/L/day]"]

            if stp_value > best_stp_value:
                best_stp_value = stp_value
                best_stp_profile = profile
                best_stp_economics = economics

            if i == total or i % max(1, total // 20) == 0:
                print(
                    f"\rGrid search: {i}/{total} ({i / total:.0%})",
                    end=""
                )

        print("\nGrid search complete.")

        results = pd.DataFrame(records)

        ranked_stp = results.sort_values(
            "Maximum space-time profit [$/L/day]",
            ascending=False
        ).reset_index(drop=True)

        ranked_stp.insert(
            0, "STP Rank", np.arange(1, len(ranked_stp) + 1)
        )

        ranked_profit = results.sort_values(
            "Maximum profit [$/L]",
            ascending=False
        ).reset_index(drop=True)

        ranked_profit.insert(
            0, "Profit Rank", np.arange(1, len(ranked_profit) + 1)
        )

        best_stp = ranked_stp.iloc[0].to_dict()
        best_profit = ranked_profit.iloc[0].to_dict()

        # Recalculate the best STP profile for figure generation.
        _, best_stp_profile, best_stp_economics = self._evaluate_condition(
            best_stp["Glucose [mM]"],
            best_stp["Glutamine [mM]"],
            best_stp["Initial VCD [cells/mL]"]
        )

        mean_rmse = accuracy.iloc[0]["Mean mAb RMSE [mg/L]"]

        uncertainty = self._simulate_economic_uncertainty(
            best_stp_profile, best_stp, mean_rmse
        )
        self._generate_figures(
            best_stp,
            best_stp_profile,
            best_stp_economics,
            mean_rmse,
            uncertainty
        )

        self._save_results(
            ranked_stp,
            accuracy,
            validation_batches,
            (glucose_bounds, glutamine_bounds, vcd_bounds),
            best_stp_profile,
            mean_rmse,
            best_stp,
            uncertainty
        )

        # Print summary.
        print("=" * 65)
        print("ECONOMIC ANALYSIS RESULTS")
        print("=" * 65)

        print(f"\nSelected model: {self.model_type}")
        print(f"NVAR lag order: p={self.p}")
        print(f"Best parameters: {self.best_parameters}")

        print("\nMODEL ACCURACY")
        print("-" * 50)
        print(
            f"Validation batches: "
            f"{int(accuracy.iloc[0]['Number of validation batches'])}"
        )
        print(
            f"Mean mAb RMSE: "
            f"{accuracy.iloc[0]['Mean mAb RMSE [mg/L]']:.3f} mg/L"
        )

        std_rmse = accuracy.iloc[0]["SD mAb RMSE [mg/L]"]
        if pd.notna(std_rmse):
            print(f"RMSE standard deviation: {std_rmse:.3f} mg/L")
        else:
            print("RMSE standard deviation: unavailable")

        print(f"\nGrid combinations evaluated: {total}")

        print("\nBEST CONDITIONS FOR MAXIMUM SPACE-TIME PROFIT")
        print("-" * 50)
        print(f"Glucose: {best_stp['Glucose [mM]']:.2f} mM")
        print(f"Glutamine: {best_stp['Glutamine [mM]']:.2f} mM")
        print(
            f"Initial VCD: "
            f"{best_stp['Initial VCD [cells/mL]']:,.0f} cells/mL"
        )
        print(
            f"Harvest time: "
            f"{best_stp['Harvest time for maximum STP [d]']:.2f} days"
        )
        print(
            f"mAb: {best_stp['mAb at maximum STP [mg/L]']:.2f} mg/L"
        )
        print(
            f"Profit at this harvest time: "
            f"${best_stp['Profit at maximum STP [$/L]']:,.2f}/L"
        )
        print(
            f"Maximum space-time profit: "
            f"${best_stp['Maximum space-time profit [$/L/day]']:,.2f}/L/day"
        )
        harvest_index = int(np.argmin(np.abs(
            self.time_days - best_stp["Harvest time for maximum STP [d]"]
        )))
        print(f"Cell culture cost: ${best_stp['Cell culture cost [$/L]']:,.2f}/L")
        print(
            "Cell culture value at harvest (95% PI): "
            f"${best_stp['Cell culture value at maximum STP [$/L]']:,.2f}/L "
            f"[${uncertainty['value_lower'][harvest_index]:,.2f}, "
            f"${uncertainty['value_upper'][harvest_index]:,.2f}]/L"
        )
        print(
            "Space-time profit (95% PI): "
            f"${best_stp['Maximum space-time profit [$/L/day]']:,.2f}/(L·d) "
            f"[${uncertainty['stp_lower'][harvest_index]:,.2f}, "
            f"${uncertainty['stp_upper'][harvest_index]:,.2f}]/(L·d)"
        )
        print(
            "mAb space-time yield (95% PI): "
            f"{best_stp['Space-time yield [mg/L/day]']:.3f} mg/(L·d) "
            f"[{uncertainty['sty_lower'][harvest_index]:.3f}, "
            f"{uncertainty['sty_upper'][harvest_index]:.3f}] mg/(L·d)"
        )

        print("\nBEST CONDITIONS FOR MAXIMUM TOTAL PROFIT")
        print("-" * 50)
        print(f"Glucose: {best_profit['Glucose [mM]']:.2f} mM")
        print(f"Glutamine: {best_profit['Glutamine [mM]']:.2f} mM")
        print(
            f"Initial VCD: "
            f"{best_profit['Initial VCD [cells/mL]']:,.0f} cells/mL"
        )
        print(
            f"Harvest time: "
            f"{best_profit['Harvest time for maximum profit [d]']:.2f} days"
        )
        print(
            f"Maximum profit: "
            f"${best_profit['Maximum profit [$/L]']:,.2f}/L"
        )

        print("\n" + "=" * 65)

        return {
            "ranked_by_space_time_profit": ranked_stp,
            "ranked_by_profit": ranked_profit,
            "model_accuracy": accuracy,
            "validation_batches": validation_batches,
            "best_space_time_profit_conditions": best_stp,
            "best_profit_conditions": best_profit
        }
=======
# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")
>>>>>>> 0bc8a4725e751c81d6c0027649197985d389a77a
