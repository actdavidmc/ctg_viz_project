import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_dotplot(df: pd.DataFrame, columna: str, grupo: str) -> None:
    """Genera un dot plot comparando grupos."""
    plt.figure(figsize=(10, 5))
    sns.stripplot(
        data=df,
        x=grupo,
        y=columna,
        jitter=False,
        size=8,
        palette="viridis"
    )
    plt.title(f"Dot Plot de {columna} por {grupo}")
    plt.grid(alpha=0.3)
    plt.show()