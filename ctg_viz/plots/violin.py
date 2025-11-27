import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_violin(
    df: pd.DataFrame,
    columna: str,
    grupo: str
) -> None:
    """Genera un violín con overlay de swarmplot.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna (str): Variable numérica.
        grupo (str): Variable categórica.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=grupo, y=columna, inner=None, palette="light:g")
    sns.swarmplot(data=df, x=grupo, y=columna, color="k", size=3)
    plt.title(f"Gráfico de Violín con Swarmplot: {columna} por {grupo}")
    plt.grid(alpha=0.3)
    plt.show()