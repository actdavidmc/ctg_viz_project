import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_boxplot(
    df: pd.DataFrame,
    columna: str,
    grupo: Optional[str] = None,
    facet: bool = False
) -> None:
    """Genera boxplots individuales o facetados por clase objetivo.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna (str): Variable numérica.
        grupo (str, optional): Variable categórica para agrupar.
        facet (bool): Si True, genera un subgráfico por categoría.
    """
    if facet and grupo:
        g = sns.FacetGrid(df, col=grupo, col_wrap=3)
        g.map_dataframe(sns.boxplot, y=columna)
        g.set_titles(col_template="{col_name}")
        plt.suptitle(f"Boxplot de {columna} por {grupo}", y=1.03)
        plt.show()
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=grupo if grupo else None, y=columna, hue=grupo)
    plt.title(f"Boxplot de {columna}" + (f" por {grupo}" if grupo else ""))
    plt.grid(alpha=0.3)
    plt.show()