import pandas as pd
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # evita abrir ventanas al graficar

from ctg_viz.plots.histograms import plot_histogram
from ctg_viz.plots.boxplots import plot_boxplot
from ctg_viz.plots.barplots import plot_barplot
from ctg_viz.plots.density import plot_density
from ctg_viz.plots.heatmap import plot_heatmap
from ctg_viz.plots.lineplots import plot_line
from ctg_viz.plots.dotplots import plot_dotplot
from ctg_viz.plots.violin import plot_violin


@pytest.fixture
def df_graf():
    np.random.seed(123)
    return pd.DataFrame({
        "x": np.random.normal(10, 2, 100),
        "y": np.random.normal(5, 1, 100),
        "cat": np.random.choice(["A", "B", "C"], size=100)
    })


def test_histogram(df_graf):
    plot_histogram(df_graf, columnas=["x"], kde=True, hue="cat", facet=True)


def test_boxplot(df_graf):
    plot_boxplot(df_graf, columna="x", grupo="cat", facet=True)


def test_barplot(df_graf):
    plot_barplot(df_graf, columna="cat")


def test_density(df_graf):
    plot_density(df_graf, columnas=["x", "y"], hue=None)


def test_heatmap(df_graf):
    plot_heatmap(df_graf, metodo="pearson", annot=False)


def test_lineplot(df_graf):
    plot_line(df_graf, columna="x")


def test_dotplot(df_graf):
    plot_dotplot(df_graf, columna="x", grupo="cat")


def test_violin(df_graf):
    plot_violin(df_graf, columna="x", grupo="cat")