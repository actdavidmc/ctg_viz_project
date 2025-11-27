import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional


def plot_barplot(
    df: pd.DataFrame,
    columna: str,
    hue: Optional[str] = None
) -> None:
    """Genera un barplot para una variable categórica con agrupación opcional.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna (str): Columna categórica.
        hue (str, optional): Columna categórica adicional para agrupar.
    """
    plt.figure(figsize=(8, 5))

    if hue:
        sns.countplot(data=df, y=columna, hue=hue, palette="viridis")
    else:
        conteo = df[columna].value_counts().sort_values(ascending=False)
        sns.barplot(x=conteo.values, y=conteo.index, palette="viridis")

    plt.title(f"Distribución de {columna}")
    plt.grid(alpha=0.3)
    plt.show()