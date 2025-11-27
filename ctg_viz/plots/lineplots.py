import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional


def plot_line(
    df: pd.DataFrame,
    columna: str,
    tiempo: Optional[str] = None
) -> None:
    """Genera una línea simulando serie temporal.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna (str): Columna numérica a graficar.
        tiempo (str, optional): Columna para usar como eje temporal.
    """
    df_temp = df.copy()

    if tiempo is None:
        df_temp["tiempo"] = range(len(df_temp))
        tiempo = "tiempo"

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_temp, x=tiempo, y=columna)
    plt.title(f"Línea temporal de {columna}")
    plt.grid(alpha=0.3)
    plt.show()