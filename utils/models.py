import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from time import time
from joblib import dump, load
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import root_mean_squared_error, r2_score, root_mean_squared_log_error

from .pipeline import create_pipeline, Pipeline
from ._config import *
from . import save_figure

#-------------------------------------------------------------------------------------------------------------------------

def load_model(model_name: str) -> BaseEstimator:
    """
    Loads a model from the models directory in the project root.

    Parameters:
        model_name (str): The name of the model to load.

    Returns:
        BaseEstimator: The loaded model.
    """
    path = f'models/{model_name}.joblib'
    if not os.path.exists(path):
        path = f"models/{model_name.lower().replace(' ', '_')}.joblib"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {model_name} not found.")

    return load(path)


def fit_tune_predict_visualize(
        model_name: str,
        model: BaseEstimator,
        df: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        gscv_param_grid: dict = None,
        rscv_param_dist: dict = None,
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
        n_iter: int = 50,
        verbose: bool = True
        ) -> BaseEstimator:
    """
    Fits the model through a pipeline, tunes the hyperparameters through gridsearch and/or randomsearch, predicts on the test set and visualizes the results.

    These visualizations consist of:
        - A table containing the scoring metrics
        - A scatterplot of the residuals
        - A scatterplot of the predicted vs. actual values
        - A histogram of the residuals
        - A learning curve of the model
        - A comparison between the test scores before and after tuning

    The visualizations are saved in the 'figure' folder. A copy of the fitted, tuned model is saved in the 'models' folder.

    Parameters:
        model_name (str): The name of the model.
        model (BaseEstimator): The model to fit.
        df (pd.DataFrame): The dataframe containing the data (only used for determining the columns to transform, not used for fitting data).
        X_train (pd.DataFrame): The training features.
        X_test (pd.DataFrame): The test features.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        gscv_param_grid (dict): The parameter grid for gridsearch. Default is None.
        rscv_param_dist (dict): The parameter distribution for randomsearch. Default is None.
        cv (int): The number of cross-validation folds. Default is 5.
        scoring (str): The scoring metric to use. Default is 'r2'.
        n_jobs (int): The number of jobs to run in parallel. Default is -1 (all available cores).
        n_iter (int): Thee number of parameter settings that are sampled. Default is 50.
        verbose (bool): Whether to print the results. Default is True.
    """
    global_start_time = time()

    if verbose:
        print(f"\n{model_name} Model:")
        print("=" * 50)

    # Create a pipeline with the model
    pipeline = create_pipeline(df, model=model);

    # Fit the pipeline on the training data and predict for comparison
    start = time()
    _ = pipeline.fit(X_train, y_train);
    end = time()
    if verbose:
        print(f"{'Fitting Time:': <30} {end - start:.2f} seconds")

    start = time()
    y_pred_0 = pipeline.predict(X_test);
    end = time()
    if verbose:
        print(f"{'Predicting Time:': <30} {end - start:.2f} seconds")
        print(f"Predicted y values:\n{y_pred_0.tolist()[:5]}\nActual y values:\n{y_test.tolist()[:5]}\n")

    # Tune the hyperparameters through gridsearch
    if gscv_param_grid:
        if verbose:
            print("GridSearchCV")
            print('-' * 50)
        start = time()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid = gscv_param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        _ = grid_search.fit(X_train, y_train);
        end = time()
        gscv_best_estimator = grid_search.best_estimator_
        gscv_best_score = grid_search.best_score_
        gscv_time = end - start
        
        if verbose:
            print(f"{'Time:': <30} {gscv_time:.2f} seconds")
            print(f"{'Best Score:': <30} {gscv_best_score:.4f}")
            print(f"{'Best Params:': <30} {grid_search.best_params_}\n")

    # Tune the hyperparameters through randomsearch
    if rscv_param_dist:
        if verbose:
            print("RandomSearchCV")
            print('-' * 50)
        start = time()
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=rscv_param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
        )
        _ = random_search.fit(X_train, y_train);
        end = time()
        rscv_best_estimator = random_search.best_estimator_
        rscv_best_score = random_search.best_score_
        rscv_time = end - start

        if verbose:
            print(f"{'Time:': <30} {rscv_time:.2f} seconds")
            print(f"{'Best Score:': <30} {rscv_best_score:.4f}")
            print(f"{'Best Params:': <30} {random_search.best_params_}\n")

    # Choose the best estimator
    if gscv_param_grid and rscv_param_dist:
        total_time = gscv_time + rscv_time
        if gscv_best_score > rscv_best_score:
            best_estimator = gscv_best_estimator

            if verbose:
                print(f"Picked GridSearchCV ({gscv_best_score:.4f}) over RandomSearchCV ({rscv_best_score:.4f})\n")
        else:
            best_estimator = rscv_best_estimator

            if verbose:
                print(f"Picked RandomSearchCV ({rscv_best_score:.4f}) over GridSearchCV ({gscv_best_score:.4f})\n")
    elif gscv_param_grid:
        total_time = gscv_time
        best_estimator = gscv_best_estimator

        if verbose:
            print(f"Picked GridSearchCV ({gscv_best_score:.4f}) as RandomSearchCV was not specified\n")
    elif rscv_param_dist:
        total_time = rscv_time
        best_estimator = rscv_best_estimator

        if verbose:
            print(f"Picked RandomSearchCV ({rscv_best_score:.4f}) as GridSearchCV was not specified\n")
    else:
        best_estimator = pipeline

        if verbose:
            print(f"Picked the original model as no tuning was specified\n")

    # Predict on the test set for the best estimator
    start = time()
    y_pred_1 = best_estimator.predict(X_test);
    end = time()

    if verbose:
        print('-' * 50)
        print(f"{'Predicting Time:': <30} {end - start:.2f} seconds")
        print(f"Predicted y values:\n{y_pred_1.tolist()[:5]}\nActual y values:\n{y_test.tolist()[:5]}\n")

    # Create the visualizations
    _compare_scores(model_name, y_pred_0, y_pred_1, y_test, verbose)
    _evaluate_model(model_name, total_time, y_pred_1, y_test, verbose)
    _plot_residuals(model_name, y_pred_1, y_test)
    _plot_predicted_vs_actual(model_name, y_pred_1, y_test)
    _plot_learning_curve(model_name, best_estimator, X_train, y_train, cv, scoring, n_jobs, verbose)
    _save_model(best_estimator, model_name, verbose=True)

    if verbose:
        print("Process Finished")
        print('-' * 50)
        print(f"{'Total Time:': <30} {(time() - global_start_time):.2f} seconds\n")
        print("=" * 50)

    return best_estimator


def _save_model(model: BaseEstimator, model_name: str, model_dir_path: str = 'models', verbose: bool = True):
    """
    Saves the model to a joblib file in the 'models' directory.

    Parameters
        model (BaseEstimator): The model to save.
        model_name (str): The name of the model.
        model_dir_path (str): The path to the directory to save the model to. Defaults to 'models'.
    """
    model_dir = Path(model_dir_path)
    model_dir.mkdir(exist_ok=True)

    path_to_model = model_dir / f"{model_name.lower().replace(' ', '_')}.joblib"
    dump(model, path_to_model)

    if verbose:
        print('-' * 50)
        print(f"Model saved as: {path_to_model}\n")


def _plot_learning_curve(model_name: str, model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5, scoring: str = 'r2', n_jobs: int = -1, verbose: bool = True):
    """
    Plots the learning curve for the model.

    Parameters
        model_name (str): The name of the model.
        model (BaseEstimator): The model to plot the learning curve for.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        cv (int): The number of cross-validation folds. Default is 5.
        scoring (str): The scoring metric to use. Default is 'r2'.
        n_jobs (int): The number of jobs to run in parallel. Default is -1.
        verbose (bool): Whether to print the results. Default is True.
    """
    if verbose:
        print("Learning Curve Results:")
        print('-' * 50)

    # Create the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    # Calculate the mean and standard deviation of the training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    if verbose:
        print(f"{'Train Mean:': <30}\n{train_mean}")
        print(f"{'Train Std:': <30}\n{train_std}")
        print(f"{'Test Mean:': <30}\n{test_mean}")
        print(f"{'Test Std:': <30}\n{test_std}\n")

    # Plot the learning curve
    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_mean, color=COLORS[0], label='Training score')
    plt.plot(train_sizes, test_mean, color=COLORS[1], label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=COLORS[0])
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color=COLORS[1])
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title(f'{model_name} Learning Curve')
    plt.legend()
    save_figure(plt, f"{model_name.lower().replace(' ', '_')}_learning_curve", subfolder='models', subsubfolder=model_name.lower())
    plt.close();


def _plot_predicted_vs_actual(model_name: str, y_pred: np.ndarray, y_test: pd.Series):
    """
    Plots the predicted values vs the actual values in a scatter plot with a regression line.

    Parameters
        model_name (str): The name of the model.
        y_pred (np.ndarray): The predicted values.
        y_test (pd.Series): The actual values.
    """
    # Concat the values into a DataFrame
    df = pd.concat([y_test, pd.Series(y_pred)], axis=1, keys=['actual', 'predicted'])

    plt.figure(figsize=(8,6))
    sns.regplot(
        data=df,
        x='actual',
        y='predicted',
        scatter_kws=dict(color=COLORS[0], s=10, alpha=.5),
        line_kws=dict(color=COLORS[1], linewidth=2),
    )
    plt.title(f"{model_name} Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    save_figure(plt, f"{model_name.lower().replace(' ', '_')}_predicted_vs_actual", subfolder='models', subsubfolder=model_name.lower())
    plt.close();


def _plot_residuals(model_name: str, y_pred: np.ndarray, y_test: pd.Series):
    """
    Plots the residuals of the model. Both as a scatter plot and as a histogram.

    Parameters:
        model_name (str): The name of the model.
        y_pred (np.ndarray): The predicted values.
        y_test (pd.Series): The actual values.
    """
    # Compute residuals
    res = y_test - y_pred

    # Plot the residuals
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x=y_pred,
        y=res,
        hue=y_pred,
        alpha=.5,
        palette=COLORS[0],
    )
    plt.axhline(0, color=COLORS[1], linestyle='--', linewidth=1)
    plt.title(f"{model_name} Residuals")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    save_figure(plt, f"{model_name.lower().replace(' ', '_')}_residuals_plot", subfolder='models', subsubfolder=model_name.lower())
    plt.close();

    # Plot the residuals histogram
    plt.figure(figsize=(8,6))
    sns.histplot(res, bins=30, kde=True, color=COLORS[0])
    plt.title(f"{model_name} Distribution of Residuals")
    plt.xlabel("Residuals")
    save_figure(plt, f"{model_name.lower().replace(' ', '_')}_residuals_distribution", subfolder='models', subsubfolder=model_name.lower())
    plt.close();


def _evaluate_model(model_name: str, total_time: int, y_pred: np.ndarray, y_test: pd.Series, verbose: bool = True):
    """
    Evaluates the model by computing the scores and visualizing the results in a table.

    Parameters:
        model_name (str): The name of the model.
        total_time (int): The total time taken to tune the model.
        y_pred (np.ndarray): The predicted values.
        y_test (pd.Series): The actual values.
        verbose (bool, optional): Whether to print the evaluation results. Defaults to True.
    """
    # Compute the scores
    scores = __compute_scores(y_pred, y_test)

    if verbose:
        print("Model Evaluation Results:")
        print("-" * 50)
        print(f"{'RMSLE:': <20} {scores['rmsle']:.4f}")
        print(f"{'RMSE:': <20} {scores['rmse']:.4f}")
        print(f"{'R2 Score:': <20} {scores['r2']:.4f}\n")

    metrics_df = pd.DataFrame({
        "Metric": ["Model Name", "Root Mean Squared Error (RMSE)", "Root Mean Squared Logarithmic Error (RMSLE)", "R² Score", "Total Time (s)"],
        "Score": [model_name, scores["rmse"], scores["rmsle"], scores["r2"], total_time]
    })

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
    save_figure(fig, f"{model_name.lower().replace(' ', '_')}_scores_table", subfolder='models', subsubfolder=model_name.lower())
    plt.close();


def _compare_scores(model_name: str, prev_pred: np.ndarray, post_pred: np.ndarray, y_test: pd.Series, verbose: bool = True) -> None:
    """
    Compares the scores of the model before and after tuning by means of a combined bar chart.

    Parameters:
        model_name (str): The name of the model.
        prev_pred (np.ndarray): The predictions before tuning.
        post_pred (np.ndarray): The predictions after tuning.
        y_test (pd.Series): The true values.
        verbose (bool, optional): Whether to print the results. Defaults to True.
    """
    # Compute the scores
    before = __compute_scores(prev_pred, y_test)
    after = __compute_scores(post_pred, y_test)

    df = pd.DataFrame([
        {'Metric': metric, 'Value': value, 'Stage': 'Before Tuning'}
        for metric, value in before.items()
    ] + [
        {'Metric': metric, 'Value': value, 'Stage': 'After Tuning'}
        for metric, value in after.items()
    ])

    if verbose:
        print("Hyperparameter Tuning Results:")
        print("-" * 50)
        print("Before Tuning:")
        print(f"{'RMSLE:': <20} {before['rmsle']:.4f}")
        print(f"{'RMSE:': <20} {before['rmse']:.4f}")
        print(f"{'R2 Score:': <20} {before['r2']:.4f}")
        print("-" * 50)
        print("After Tuning:")
        print(f"{'RMSLE:': <20} {after['rmsle']:.4f}")
        print(f"{'RMSE:': <20} {after['rmse']:.4f}")
        print(f"{'R2 Score:': <20} {after['r2']:.4f}\n")

    # Create the bar chart
    g = sns.catplot(
        data=df,
        x='Stage',
        y='Value',
        hue='Stage',
        col='Metric',
        kind='bar',
        orient='v',
        palette=COLORS[:2],
        legend=False,
        height=4,
        aspect=0.8,
        sharey=False,
    )
    g.set_titles('{col_name}')
    g.set_axis_labels('', 'Score')
    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle(f'{model_name} Scores Before and After Tuning')
    save_figure(g.figure, f"{model_name.lower().replace(' ', '_')}_tuning_comparison", subfolder='models', subsubfolder=model_name.lower())
    plt.close();


def __compute_scores(y_pred: np.ndarray, y_test: pd.Series) -> dict:
    """
    Computes the scores of the model performance.

    Parameters:
        y_pred (np.ndarray): The predicted values.
        y_test (pd.Series): The true values.

    Returns:
        dict: The scores.
    """
    scores = {}

    # Alter the y_pred to always be positive for the RMSLE
    y_pred_alt = np.where(y_pred < 0, 0, y_pred)

    scores['rmsle'] = root_mean_squared_log_error(y_test, y_pred_alt)
    scores['rmse'] = root_mean_squared_error(y_test, y_pred)
    scores['r2'] = r2_score(y_test, y_pred)

    return scores