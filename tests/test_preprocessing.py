import pandas as pd
import numpy as np
import pytest

from ctg_viz.preprocessing import (
    estandarizar_columnas,
    convertir_tipos,
    eliminar_col_nulos,
    imputar_valores,
    outliers_iqr,
    outliers_z,
    tratar_outliers,
    preprocessing
)


@pytest.fixture
def df_robusto():
    np.random.seed(42)

    df = pd.DataFrame({
        " Col A ": np.append(np.random.normal(50, 10, 95), [300, 400, 500, np.nan, np.nan]),
        "B-Col": np.append(np.random.uniform(0, 1, 97), [5, 6, np.nan]),
        "Categoria": np.random.choice(["A", "B", "C", None], size=100),
        "Constante": 1
    })

    return df


def test_estandarizar_columnas(df_robusto):
    df = estandarizar_columnas(df_robusto)
    assert "col_a" in df.columns
    assert "b_col" in df.columns
    assert isinstance(df, pd.DataFrame)


def test_convertir_tipos(df_robusto):
    df = convertir_tipos(df_robusto)
    assert isinstance(df, pd.DataFrame)
    assert df["Constante"].dtype.name in ("Int64", "int64", "float64")


def test_eliminar_col_nulos(df_robusto):
    df = eliminar_col_nulos(df_robusto, threshold=0.10)
    # Categoria tiene más de 25% nulos → debe eliminarse
    assert "Categoria" not in df.columns


def test_imputar_mean(df_robusto):
    df = imputar_valores(df_robusto, metodo="mean")
    assert df.isna().sum().sum() == 0


def test_imputar_median(df_robusto):
    df = imputar_valores(df_robusto, metodo="median")
    assert df.isna().sum().sum() == 0


def test_imputar_knn(df_robusto):
    df = imputar_valores(df_robusto, metodo="knn", knn_vecinos=3)
    assert df.isna().sum().sum() == 0


def test_outliers_iqr(df_robusto):
    resultado = outliers_iqr(df_robusto)
    assert isinstance(resultado, dict)
    assert all(isinstance(v, pd.Series) for v in resultado.values())


def test_outliers_z(df_robusto):
    resultado = outliers_z(df_robusto, umbral=3)
    assert isinstance(resultado, dict)
    assert all(isinstance(v, pd.Series) for v in resultado.values())


def test_tratar_outliers_winsorize(df_robusto):
    df = tratar_outliers(df_robusto, metodo="iqr", accion="winsorize")
    assert isinstance(df, pd.DataFrame)
    # Debe seguir teniendo el mismo número de filas
    assert len(df) == len(df_robusto)


def test_tratar_outliers_drop(df_robusto):
    df = tratar_outliers(df_robusto, metodo="zscore", accion="drop")
    assert len(df) < len(df_robusto)


def test_preprocessing_pipeline(df_robusto, tmp_path):
    # Guardamos un CSV temporal
    path = tmp_path / "data.csv"
    df_robusto.to_csv(path, index=False)

    df_final = preprocessing(path, metodo_imputacion="mean", metodo_outliers="iqr")
    assert isinstance(df_final, pd.DataFrame)
    assert df_final.isna().sum().sum() == 0