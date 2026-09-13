
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib_venn import venn3

from utils.anomaly_detection import (
    detect_anomalies_iqr,
    detect_anomalies_isolation_forest,
    build_comparison_table,
    create_pca_projection
)


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Détection des anomalies",
    page_icon="🔎",
    layout="wide"
)


# ==========================================================
# TITRE
# ==========================================================

st.title("🔎 Détection des anomalies")

st.markdown(
    """
    Cette page permet de détecter et comparer les anomalies
    à l'aide de trois méthodes :

    - **IQR** : détection univariée basée sur les quartiles ;
    - **Isolation Forest univarié** ;
    - **Isolation Forest multivarié**.

    Les résultats sont ensuite comparés afin d'identifier
    les observations détectées par plusieurs méthodes.
    """
)


# ==========================================================
# RECUPERATION DU DATASET DE LA PAGE 2
# ==========================================================

if "df_anomaly" not in st.session_state:

    st.warning(
        "⚠️ Aucun jeu de données propre n'est disponible."
    )

    st.info(
        "Veuillez d'abord aller sur la page "
        "'Description' et créer le jeu de données "
        "sans valeurs manquantes."
    )

    st.stop()


df = st.session_state["df_anomaly"].copy()


# ==========================================================
# VARIABLES DISPONIBLES
# ==========================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()


if not numeric_cols:

    st.error(
        "❌ Aucune variable quantitative disponible."
    )

    st.stop()


# ==========================================================
# INFORMATIONS DATASET
# ==========================================================

st.header("1️⃣ Jeu de données utilisé")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Observations",
        len(df)
    )

with col2:
    st.metric(
        "Variables quantitatives",
        len(numeric_cols)
    )

with col3:
    st.metric(
        "Valeurs manquantes",
        int(df.isna().sum().sum())
    )


st.dataframe(
    df,
    use_container_width=True,
    height=300
)


# ==========================================================
# PARAMETRES
# ==========================================================

st.header("2️⃣ Paramétrage de la détection")


# ----------------------------------------------------------
# VARIABLE D'INTERET
# ----------------------------------------------------------

variable_of_interest = st.selectbox(
    "🎯 Variable d'intérêt",
    options=numeric_cols
)


# ----------------------------------------------------------
# VARIABLES MULTIVARIEES
# ----------------------------------------------------------

default_multi = [
    col
    for col in numeric_cols
    if col != variable_of_interest
]

selected_multi = st.multiselect(
    "📌 Variables utilisées pour Isolation Forest multivarié",
    options=numeric_cols,
    default=default_multi
)


# ----------------------------------------------------------
# PARAMETRE IQR
# ----------------------------------------------------------

iqr_k = st.slider(
    "Coefficient k de l'IQR",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help=(
        "1.5 correspond à la convention classique. "
        "Augmenter k rend la détection plus restrictive."
    )
)


# ----------------------------------------------------------
# CONTAMINATION
# ----------------------------------------------------------

contamination = st.slider(
    "Proportion d'anomalies attendue",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help=(
        "Proportion d'observations considérées comme "
        "potentiellement anormales par Isolation Forest."
    )
)


# ==========================================================
# LANCEMENT
# ==========================================================

st.divider()

run_detection = st.button(
    "🚀 Lancer la détection des anomalies",
    type="primary"
)


if run_detection:

    if len(selected_multi) < 2:

        st.error(
            "Sélectionnez au moins deux variables "
            "pour l'Isolation Forest multivarié."
        )

        st.stop()


    # ======================================================
    # IQR
    # ======================================================

    result_iqr, iqr_info = (
        detect_anomalies_iqr(
            df,
            variable_of_interest,
            k=iqr_k
        )
    )


    # ======================================================
    # ISOLATION FOREST UNIVARIE
    # ======================================================

    result_uni, model_uni = (
        detect_anomalies_isolation_forest(
            df,
            feature_cols=[
                variable_of_interest
            ],
            contamination=contamination
        )
    )


    # ======================================================
    # ISOLATION FOREST MULTIVARIE
    # ======================================================

    result_multi, model_multi = (
        detect_anomalies_isolation_forest(
            df,
            feature_cols=selected_multi,
            contamination=contamination
        )
    )


    # ======================================================
    # TABLEAU CONSOLIDE
    # ======================================================

    comparison = build_comparison_table(
        result_iqr,
        result_uni,
        result_multi,
        variable_of_interest
    )


    # Ajout des informations originales

    if "battery" in df.columns:

        comparison["battery"] = (
            df["battery"]
        )

    if "cycle_index" in df.columns:

        comparison["cycle_index"] = (
            df["cycle_index"]
        )

    if "type" in df.columns:

        comparison["type"] = (
            df["type"]
        )


    # Sauvegarde dans session_state

    st.session_state[
        "anomaly_comparison"
    ] = comparison

    st.session_state[
        "anomaly_result_iqr"
    ] = result_iqr

    st.session_state[
        "anomaly_result_uni"
    ] = result_uni

    st.session_state[
        "anomaly_result_multi"
    ] = result_multi

    st.session_state[
        "anomaly_multi_variables"
    ] = selected_multi

    st.session_state[
        "anomaly_variable_of_interest"
    ] = variable_of_interest

    st.session_state[
        "iqr_information"
    ] = iqr_info

    st.success(
        "✅ Détection terminée."
    )


# ==========================================================
# AFFICHAGE DES RESULTATS
# ==========================================================

if "anomaly_comparison" not in st.session_state:

    st.info(
        "👆 Configurez les paramètres puis cliquez sur "
        "'Lancer la détection des anomalies'."
    )

    st.stop()


comparison = st.session_state[
    "anomaly_comparison"
]

result_multi = st.session_state[
    "anomaly_result_multi"
]

selected_multi = st.session_state[
    "anomaly_multi_variables"
]

variable_of_interest = st.session_state[
    "anomaly_variable_of_interest"
]

iqr_info = st.session_state[
    "iqr_information"
]


# ==========================================================
# RESUME
# ==========================================================

st.header("3️⃣ Résultats de la détection")


iqr_count = int(
    comparison["anomaly_iqr"].sum()
)

uni_count = int(
    comparison[
        "anomaly_isoforest_uni"
    ].sum()
)

multi_count = int(
    comparison[
        "anomaly_isoforest_multi"
    ].sum()
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Anomalies IQR",
        iqr_count
    )

with col2:
    st.metric(
        "IsoForest univarié",
        uni_count
    )

with col3:
    st.metric(
        "IsoForest multivarié",
        multi_count
    )

with col4:
    st.metric(
        "Détectées par ≥ 2 méthodes",
        int(
            (
                comparison[
                    "n_methods_flagged"
                ] >= 2
            ).sum()
        )
    )


# ==========================================================
# IQR
# ==========================================================

st.subheader(
    "📦 3.1 Détection IQR"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Q1",
        f"{iqr_info['q1']:.4f}"
    )

with col2:
    st.metric(
        "Q3",
        f"{iqr_info['q3']:.4f}"
    )

with col3:
    st.metric(
        "Borne basse",
        f"{iqr_info['lower_bound']:.4f}"
    )

with col4:
    st.metric(
        "Borne haute",
        f"{iqr_info['upper_bound']:.4f}"
    )


# Graphique IQR avec Plotly

normal_iqr = comparison[
    comparison["anomaly_iqr"] == 0
]

anomaly_iqr = comparison[
    comparison["anomaly_iqr"] == 1
]

fig_iqr = plt.figure(
    figsize=(12, 5)
)

ax = fig_iqr.add_subplot(111)

ax.scatter(
    normal_iqr.index,
    normal_iqr[variable_of_interest],
    alpha=0.5,
    s=20,
    label="Normal"
)

ax.scatter(
    anomaly_iqr.index,
    anomaly_iqr[variable_of_interest],
    s=40,
    label="Anomalie"
)

ax.axhline(
    iqr_info["lower_bound"],
    linestyle="--",
    label="Borne IQR"
)

ax.axhline(
    iqr_info["upper_bound"],
    linestyle="--"
)

ax.set_title(
    f"Anomalies IQR : {iqr_count} / {len(df)}"
)

ax.set_xlabel("Index")
ax.set_ylabel(variable_of_interest)

ax.legend()

st.pyplot(
    fig_iqr,
    use_container_width=True
)

plt.close(fig_iqr)


# ==========================================================
# ISOLATION FOREST UNIVARIE
# ==========================================================

st.subheader(
    "🌲 3.2 Isolation Forest univarié"
)

result_uni = st.session_state[
    "anomaly_result_uni"
]

fig_uni = plt.figure(
    figsize=(12, 5)
)

ax = fig_uni.add_subplot(111)

normal_uni = result_uni[
    result_uni["anomaly_isoforest"] == 0
]

anomaly_uni = result_uni[
    result_uni["anomaly_isoforest"] == 1
]

ax.scatter(
    normal_uni.index,
    normal_uni[variable_of_interest],
    alpha=0.5,
    s=20,
    label="Normal"
)

ax.scatter(
    anomaly_uni.index,
    anomaly_uni[variable_of_interest],
    s=40,
    label="Anomalie"
)

ax.set_title(
    f"Isolation Forest univarié : "
    f"{uni_count} / {len(df)} anomalies"
)

ax.set_xlabel("Index")
ax.set_ylabel(variable_of_interest)

ax.legend()

st.pyplot(
    fig_uni,
    use_container_width=True
)

plt.close(fig_uni)


# ==========================================================
# ISOLATION FOREST MULTIVARIE
# ==========================================================

st.subheader(
    "🌲 3.3 Isolation Forest multivarié"
)

st.write(
    "Variables utilisées : "
    + ", ".join(selected_multi)
)


result_plot, explained_variance = (
    create_pca_projection(
        result_multi,
        selected_multi
    )
)


fig_pca = plt.figure(
    figsize=(10, 7)
)

ax = fig_pca.add_subplot(111)

normal_multi = result_plot[
    result_plot["anomaly_isoforest"] == 0
]

anomaly_multi = result_plot[
    result_plot["anomaly_isoforest"] == 1
]

ax.scatter(
    normal_multi["PCA1"],
    normal_multi["PCA2"],
    alpha=0.5,
    s=20,
    label="Normal"
)

ax.scatter(
    anomaly_multi["PCA1"],
    anomaly_multi["PCA2"],
    s=45,
    label="Anomalie"
)

ax.set_title(
    f"Isolation Forest multivarié - projection PCA\n"
    f"{multi_count} anomalies"
)

ax.set_xlabel(
    f"PC1 ({explained_variance[0] * 100:.1f} %)"
)

ax.set_ylabel(
    f"PC2 ({explained_variance[1] * 100:.1f} %)"
)

ax.legend()

st.pyplot(
    fig_pca,
    use_container_width=True
)

plt.close(fig_pca)


# ==========================================================
# COMPARAISON DES METHODES
# ==========================================================

st.header(
    "4️⃣ Comparaison des méthodes"
)


# ==========================================================
# TABLEAU
# ==========================================================

st.subheader(
    "📋 Tableau consolidé"
)

display_columns = [
    variable_of_interest,
    "iqr_score",
    "anomaly_iqr",
    "isoforest_uni_score",
    "anomaly_isoforest_uni",
    "isoforest_multi_score",
    "anomaly_isoforest_multi",
    "n_methods_flagged"
]

# Ajouter les métadonnées si elles existent

for col in [
    "battery",
    "cycle_index",
    "type"
]:

    if col in comparison.columns:
        display_columns.insert(
            0,
            col
        )


st.dataframe(
    comparison[
        display_columns
    ].sort_values(
        "n_methods_flagged",
        ascending=False
    ),
    use_container_width=True,
    height=500
)


# ==========================================================
# VENN
# ==========================================================

st.subheader(
    "⭕ 4.1 Recouvrement des anomalies"
)

set_iqr = set(
    comparison[
        comparison["anomaly_iqr"] == 1
    ].index
)

set_uni = set(
    comparison[
        comparison[
            "anomaly_isoforest_uni"
        ] == 1
    ].index
)

set_multi = set(
    comparison[
        comparison[
            "anomaly_isoforest_multi"
        ] == 1
    ].index
)


fig_venn = plt.figure(
    figsize=(8, 8)
)

venn3(
    [
        set_iqr,
        set_uni,
        set_multi
    ],
    set_labels=[
        "IQR",
        "IsoForest univarié",
        "IsoForest multivarié"
    ]
)

plt.title(
    "Recouvrement des anomalies détectées"
)

st.pyplot(
    fig_venn,
    use_container_width=False
)

plt.close(fig_venn)


# ==========================================================
# SCATTER AGREEMENT
# ==========================================================

st.subheader(
    "🎯 4.2 Accord entre les méthodes"
)

st.markdown(
    """
    Plus une observation est détectée par plusieurs méthodes,
    plus elle constitue un candidat intéressant pour une analyse
    approfondie.
    """
)


fig_agreement = plt.figure(
    figsize=(12, 6)
)

ax = fig_agreement.add_subplot(111)

scatter = ax.scatter(
    comparison.index,
    comparison[variable_of_interest],
    c=comparison["n_methods_flagged"],
    s=35,
    vmin=0,
    vmax=3,
    edgecolors="grey",
    linewidths=0.3
)

ax.set_title(
    "Nombre de méthodes détectant chaque observation"
)

ax.set_xlabel("Index")
ax.set_ylabel(variable_of_interest)

colorbar = plt.colorbar(
    scatter,
    ax=ax
)

colorbar.set_label(
    "Nombre de méthodes"
)

colorbar.set_ticks(
    [0, 1, 2, 3]
)

st.pyplot(
    fig_agreement,
    use_container_width=True
)

plt.close(fig_agreement)


# ==========================================================
# ANOMALIES DETECTEES PAR PLUSIEURS METHODES
# ==========================================================

st.subheader(
    "🚨 Observations détectées par plusieurs méthodes"
)

multi_anomalies = comparison[
    comparison["n_methods_flagged"] >= 2
].sort_values(
    "n_methods_flagged",
    ascending=False
)


if multi_anomalies.empty:

    st.info(
        "Aucune observation n'est détectée par au moins "
        "deux méthodes."
    )

else:

    st.write(
        f"{len(multi_anomalies)} observation(s) "
        "détectée(s) par au moins deux méthodes."
    )

    st.dataframe(
        multi_anomalies,
        use_container_width=True
    )


# ==========================================================
# TELECHARGEMENT
# ==========================================================

st.subheader(
    "⬇️ Export des résultats"
)

csv = comparison.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Télécharger les résultats CSV",
    data=csv,
    file_name="resultats_detection_anomalies.csv",
    mime="text/csv"
)
