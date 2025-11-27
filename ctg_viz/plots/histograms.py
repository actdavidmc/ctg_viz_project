import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_histogram(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    bins: int = 30,
    kde: bool = True,
    hue: Optional[str] = None,
    facet: bool = False
) -> None:
    """Genera histogramas con KDE, agrupación opcional y facetado.

    Args:
        df (pd.DataFrame): DataFrame original.
        columnas (list[str], optional): Columnas numéricas a graficar.
        bins (int): Número de bins.
        kde (bool): Mostrar línea de densidad.
        hue (str, optional): Columna categórica para agrupar.
        facet (bool): Si True, genera un histograma por categoría.
    """
    if columnas is None:
        columnas = df.select_dtypes(include="number").columns.tolist()

    for col in columnas:

        if facet and hue:
            g = sns.FacetGrid(df, col=hue, col_wrap=3, sharex=False)
            g.map_dataframe(sns.histplot, x=col, kde=kde, bins=bins)
            g.set_titles(col_template="{col_name}")
            plt.suptitle(f"Histograma de {col} por {hue}", y=1.05)
            plt.show()
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(df, x=col, bins=bins, kde=kde, hue=hue, palette="viridis")
        plt.title(f"Histograma de {col}" + (f" por {hue}" if hue else ""))
        plt.grid(alpha=0.3)
        plt.show()
