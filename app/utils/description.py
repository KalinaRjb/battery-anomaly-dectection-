
import pandas as pd
import numpy as np
import plotly.express as px


def get_quantitative_variables(df):
    """
    Retourne les variables quantitatives du DataFrame.
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def missing_data_summary(df):
    """
    Produit un tableau récapitulatif des données manquantes.
    """

    summary = pd.DataFrame({
        "Variable": df.columns,
        "Type": [str(df[col].dtype) for col in df.columns],
        "Valeurs manquantes": [
            df[col].isna().sum()
            for col in df.columns
        ],
        "% manquant": [
            round(df[col].isna().mean() * 100, 2)
            for col in df.columns
        ],
        "Valeurs présentes": [
            df[col].notna().sum()
            for col in df.columns
        ],
        "Valeurs uniques": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    return summary.sort_values(
        "% manquant",
        ascending=False
    )


def create_histogram(df, variable, nbins=50):
    """
    Crée un histogramme Plotly pour une variable quantitative.
    """

    data = df[variable].dropna()

    fig = px.histogram(
        data,
        x=variable,
        nbins=nbins,
        title=f"Distribution de {variable}",
        marginal="box"
    )

    fig.update_layout(
        xaxis_title=variable,
        yaxis_title="Nombre d'observations"
    )

    return fig


def create_scatter_matrix(
    df,
    variables,
    color_variable=None,
    max_points=3000
):
    """
    Crée une matrice de nuages de points
    pour plusieurs variables quantitatives.

    max_points permet d'éviter de ralentir Streamlit
    si le DataFrame contient beaucoup de lignes.
    """

    data = df.copy()

    columns = variables.copy()

    if color_variable is not None:
        columns.append(color_variable)

    data = data[columns].dropna()

    # Échantillonnage si trop de lignes
    if len(data) > max_points:
        data = data.sample(
            max_points,
            random_state=42
        )

    fig = px.scatter_matrix(
        data,
        dimensions=variables,
        color=color_variable,
        title="Matrice des relations entre variables"
    )

    fig.update_traces(
        diagonal_visible=False
    )

    fig.update_layout(
        height=max(700, 250 * len(variables)),
        width=1200
    )

    return fig


def create_missing_bar_chart(df):
    """
    Crée un graphique du pourcentage de valeurs manquantes
    par variable.
    """

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    missing.columns = [
        "Variable",
        "% manquant"
    ]

    missing = missing[
        missing["% manquant"] > 0
    ]

    if missing.empty:
        return None

    fig = px.bar(
        missing,
        x="Variable",
        y="% manquant",
        title="Pourcentage de données manquantes par variable",
        text_auto=".1f"
    )

    fig.update_layout(
        xaxis_title="Variable",
        yaxis_title="% de valeurs manquantes",
        xaxis_tickangle=-45
    )

    return fig


def create_missing_matrix(df, max_rows=1000):
    """
    Crée une matrice visuelle des données manquantes.

    Pour éviter de ralentir l'application, on limite
    le graphique à max_rows lignes.
    """

    data = df

    if len(data) > max_rows:
        data = data.sample(
            max_rows,
            random_state=42
        )

    missing_matrix = data.isna().astype(int)

    fig = px.imshow(
        missing_matrix.T,
        aspect="auto",
        labels={
            "x": "Observation",
            "y": "Variable",
            "color": "Manquant"
        },
        title="Visualisation des données manquantes"
    )

    return fig


def create_anomaly_dataset(
    df,
    variable_of_interest,
    selected_variables
):
    """
    Crée un DataFrame sans valeurs manquantes
    pour la variable d'intérêt et les variables sélectionnées.
    """

    variables = list(
        dict.fromkeys(
            [variable_of_interest] + selected_variables
        )
    )

    data = df[variables].copy()

    rows_before = len(data)

    data_complete = data.dropna(
        subset=variables
    ).copy()

    rows_after = len(data_complete)

    information = {
        "variables": variables,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "percentage_kept": (
            round(
                rows_after / rows_before * 100,
                2
            )
            if rows_before > 0
            else 0
        )
    }

    return data_complete, information
