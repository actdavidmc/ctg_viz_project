import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(
    df: pd.DataFrame,
    metodo: str = "pearson",
    annot: bool = False,
    cmap: str = "coolwarm"
) -> None:
    """Genera un heatmap de correlaciones numéricas con Pearson o Spearman.

    Args:
        df (pd.DataFrame): DataFrame original.
        metodo (str): pearson | spearman.
        annot (bool): Mostrar valores numéricos.
        cmap (str): Paleta de colores.
    """
    corr = df.select_dtypes(include="number").corr(method=metodo)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=annot, cmap=cmap, linewidths=0.5)
    plt.title(f"Mapa de Calor (método: {metodo})")
    plt.show()