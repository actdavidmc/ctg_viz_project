import pandas as pd
import numpy as np
import pytest

from ctg_viz.categorization import (
    check_data_completeness_davidmaganacelis,
    clasificar_columnas,
    resumen_categorizacion
)


@pytest.fixture
def df_robusto():
    np.random.seed(0)
    return pd.DataFrame({
        "num1": np.append(np.random.normal(0, 1, 95), [None, None, None, 100]),
        "num2": np.random.randint(0, 5, 100),
        "cat": np.random.choice(["A", "B", "C"], 100),
        "str_col": ["texto"] * 100
    })


def test_check_data_completeness(df_robusto):
    resumen = check_data_completeness_davidmaganacelis(df_robusto)
    assert isinstance(resumen, pd.DataFrame)
    assert "n_nulls" in resumen.columns
    assert "pct_completeness" in resumen.columns


def test_clasificacion_variables(df_robusto):
    clasif = clasificar_columnas(df_robusto)
    assert isinstance(clasif, dict)
    assert "continuas" in clasif
    assert "discretas" in clasif
    assert "categoricas" in clasif


def test_resumen_categorizacion(df_robusto):
    resumen = resumen_categorizacion(df_robusto)
    assert "clasificacion" in resumen.columns
    assert resumen.shape[0] == df_robusto.shape[1]