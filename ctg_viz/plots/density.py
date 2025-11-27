import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_density(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    hue: Optional[str] = None
) -> None:
    """Genera curvas KDE para columnas numéricas, agrupadas opcionalmente.

    Args:
        df (pd.DataFrame): DataFrame original.
        columnas (list[str], optional): Columnas numéricas.
        hue (str, optional): Columna categórica para agrupar.
    """
    if columnas is None:
        columnas = df.select_dtypes(include="number").columns.tolist()

    plt.figure(figsize=(10, 6))

    if hue:
        sns.kdeplot(data=df, x=columnas[0], hue=hue, fill=False)
        plt.title(f"Densidad de {columnas[0]} por {hue}")
        plt.grid(alpha=0.3)
        plt.show()
        return

    # Múltiples KDE no agrupadas
    for col in columnas:
        sns.kdeplot(df[col], linewidth=2, label=col)

    plt.legend()
    plt.title("Curvas de Densidad (KDE)")
    plt.grid(alpha=0.3)
    plt.show()