import matplotlib.pyplot as plt
from pathlib import Path

#-----------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str, subfolder: str = None, dpi: int = 300, bbox_inches: str = 'tight', **kwargs):
    """
    Saves a matplotlib figure to the figures directory in the project root.

    Parameters:
        fig (plt.Figure): The figure to save.
        filename (str): The name of the file (without extension).
        subfolder (str, optional): The subfolder to save the figure in. Defaults to None.
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

    # Ensure the filename has an extension
    if not Path(filename).suffix:
        filename += ".png"

    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
