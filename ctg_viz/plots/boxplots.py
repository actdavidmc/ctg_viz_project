import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_boxplot(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    hue: Optional[str] = None
) -> None:
    """Genera boxplots con agrupación opcional.

    Args:
        df (pd.DataFrame): DataFrame original.
        columnas (list[str], optional): Columnas numéricas.
        hue (str, optional): Columna categórica para comparar grupos.
    """
    if columnas is None:
        columnas = df.select_dtypes(include="number").columns.tolist()

    if len(columnas) > 1:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[columnas], orient="h")
        plt.title("Boxplots de múltiples variables")
        plt.grid(alpha=0.3)
        plt.show()
        return

    col = columnas[0]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=col, y=hue if hue else None, hue=hue, orient="h")
    plt.title(f"Boxplot de {col}" + (f" por {hue}" if hue else ""))
    plt.grid(alpha=0.3)
    plt.show()
