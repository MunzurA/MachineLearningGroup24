import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline

#-----------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str, subfolder: str = None, subsubfolder: str = None, dpi: int = 300, bbox_inches: str = 'tight', **kwargs):
    """
    Saves a matplotlib figure to the figures directory in the project root.

    Parameters:
        fig (plt.Figure): The figure to save.
        filename (str): The name of the file (without extension).
        subfolder (str, optional): The subfolder to save the figure in. Defaults to None.
        subsubfolder (str, optional): The subsubfolder to save the figure in. Defaults to None.
        dpi (int, optional): The resolution of the saved figure. Defaults to 300.
        bbox_inches (str, optional): The bounding box setting. Defaults to 'tight'.
        **kwargs: Additional keyword arguments to pass to the plt.savefig() function.
    """
    # Create the figures directory if it doesn't exist
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Create the subfolder if it doesn't exist
    if subfolder:
        save_dir = figures_dir / subfolder
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = figures_dir

    if subsubfolder:
        save_dir = save_dir / subsubfolder
        save_dir.mkdir(exist_ok=True)

    # Ensure the filename has an extension
    if not Path(filename).suffix:
        filename += ".png"

    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


def debug_pipeline(pipeline: Pipeline, X_test: pd.DataFrame):
    """
    Debug a scikit-learn pipeline by showing transformations and NaN values at each step.

    Parameters:
        pipeline (sklearn.Pipeline): The sklearn Pipeline object to debug
        X_test (pd.DataFrame): The test data to transform
    """
    X = X_test.copy()

    print(f"Original data shape: {X.shape}")
    print(f"Original NaN count: {X.isna().sum().sum()}")

    for name, transformer in pipeline.steps:
        print(f"\n{'='*50}")
        print(f"Step: {name}")

        if name == 'model_selection':
            print("Skipping model step")
            continue

        X = transformer.transform(X)

        print(f"Transformed data shape: {X.shape}")
        print(f"Transformed NaN count: {X.isna().sum().sum()}")

        if X.isna().sum().sum() > 0:
            nan_cols = X.columns[X.isna().any()].tolist()
            print(f"Columns with NaN values: {nan_cols}")
            for col in nan_cols:
                nan_count = X[col].isna().sum()
                print(f" - {col}: {nan_count} NaNs ({nan_count/len(X)*100:.2f}%)")
