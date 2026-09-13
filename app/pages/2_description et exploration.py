
import streamlit as st
import pandas as pd
import numpy as np

from utils.description import (
    get_quantitative_variables,
    missing_data_summary,
    create_histogram,
    create_scatter_matrix,
    create_missing_bar_chart,
    create_missing_matrix,
    create_anomaly_dataset
)


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Description des données",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# TITRE
# ==========================================================

st.title("📊 Description et exploration des données")

st.markdown(
    """
    Cette page permet d'explorer le jeu de données issu du
    NASA Battery Dataset avant la détection des anomalies.

    Vous pouvez :
    - analyser la distribution des variables quantitatives ;
    - étudier les relations entre plusieurs variables ;
    - analyser les données manquantes ;
    - créer un jeu de données complet destiné à la détection
      d'anomalies.
    """
)


# ==========================================================
# RECUPERATION DU DATAFRAME
# ==========================================================

if "df_all" not in st.session_state:

    st.warning(
        "⚠️ Aucune donnée n'est disponible."
    )

    st.info(
        "Veuillez d'abord aller sur la page "
        "'Données' et charger un ou plusieurs fichiers .mat."
    )

    st.stop()


df = st.session_state["df_all"].copy()


# ==========================================================
# INFORMATIONS GENERALES
# ==========================================================

st.header("1️⃣ Vue d'ensemble du jeu de données")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Nombre de lignes",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Nombre de variables",
        len(df.columns)
    )

with col3:
    st.metric(
        "Variables quantitatives",
        len(get_quantitative_variables(df))
    )

with col4:
    st.metric(
        "Variables avec NA",
        int(df.isna().any().sum())
    )


# ==========================================================
# TYPES DES VARIABLES
# ==========================================================

with st.expander("🔎 Voir les types de variables"):

    types_df = pd.DataFrame({
        "Variable": df.columns,
        "Type": [
            str(df[col].dtype)
            for col in df.columns
        ],
        "Valeurs uniques": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    st.dataframe(
        types_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# HISTOGRAMMES
# ==========================================================

st.header("2️⃣ Distribution des variables quantitatives")

quantitative_variables = get_quantitative_variables(df)

if not quantitative_variables:

    st.warning(
        "Aucune variable quantitative n'a été trouvée."
    )

else:

    selected_hist_variables = st.multiselect(
        "Sélectionner les variables à visualiser",
        options=quantitative_variables,
        default=quantitative_variables
    )

    nbins = st.slider(
        "Nombre de classes de l'histogramme",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    if selected_hist_variables:

        for variable in selected_hist_variables:

            fig = create_histogram(
                df,
                variable,
                nbins=nbins
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Sélectionnez au moins une variable."
        )


# ==========================================================
# MATRICE DE NUAGES DE POINTS
# ==========================================================

st.header("3️⃣ Relations entre variables quantitatives")

st.markdown(
    """
    Sélectionnez plusieurs variables quantitatives pour
    analyser leurs relations deux à deux.
    """
)

selected_scatter_variables = st.multiselect(
    "Variables à comparer",
    options=quantitative_variables,
    default=quantitative_variables[:4]
)


# Variable permettant de colorer les observations

categorical_variables = df.select_dtypes(
    exclude=np.number
).columns.tolist()

color_options = ["Aucune"] + categorical_variables

color_variable = st.selectbox(
    "Variable pour colorer les observations (optionnel)",
    options=color_options
)

if len(selected_scatter_variables) >= 2:

    if color_variable == "Aucune":
        color = None
    else:
        color = color_variable

    fig = create_scatter_matrix(
        df,
        selected_scatter_variables,
        color_variable=color
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Sélectionnez au moins deux variables "
        "pour créer la matrice de nuages de points."
    )


# ==========================================================
# DONNEES MANQUANTES
# ==========================================================

st.header("4️⃣ Analyse des données manquantes")


missing_summary = missing_data_summary(df)


# Nombre total de valeurs manquantes

total_missing = int(
    df.isna().sum().sum()
)

total_cells = df.shape[0] * df.shape[1]

missing_percentage = (
    total_missing / total_cells * 100
    if total_cells > 0
    else 0
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Nombre total de valeurs manquantes",
        f"{total_missing:,}"
    )

with col2:
    st.metric(
        "% de cellules manquantes",
        f"{missing_percentage:.2f}%"
    )

with col3:
    st.metric(
        "Variables contenant des NA",
        int(df.isna().any().sum())
    )


st.subheader("📋 Détail par variable")

st.dataframe(
    missing_summary,
    use_container_width=True,
    hide_index=True
)


# Graphique des valeurs manquantes

missing_fig = create_missing_bar_chart(df)

if missing_fig is not None:

    st.plotly_chart(
        missing_fig,
        use_container_width=True
    )

else:

    st.success(
        "✅ Le DataFrame ne contient aucune donnée manquante."
    )





# ==========================================================
# PREPARATION POUR LA DETECTION D'ANOMALIES
# ==========================================================

st.header(
    "5️⃣ Préparation des données pour la détection d'anomalies"
)

st.markdown(
    """
    Sélectionnez une variable d'intérêt et les variables
    explicatives que vous souhaitez utiliser pour la détection
    d'anomalies.

    Les lignes contenant une valeur manquante dans l'une des
    variables sélectionnées seront supprimées.
    """
)


# Variable d'intérêt

variable_of_interest = st.selectbox(
    "🎯 Variable d'intérêt",
    options=quantitative_variables
)


# Variables explicatives

remaining_variables = [
    variable
    for variable in quantitative_variables
    if variable != variable_of_interest
]

selected_variables = st.multiselect(
    "📌 Variables à utiliser pour l'analyse",
    options=remaining_variables,
    default=[]
)


# ==========================================================
# CREATION DU DATASET COMPLET
# ==========================================================

if st.button(
    "🧹 Créer le jeu de données sans valeurs manquantes",
    type="primary"
):

    if not selected_variables:

        st.warning(
            "Sélectionnez au moins une variable en plus "
            "de la variable d'intérêt."
        )

    else:

        df_anomaly, information = create_anomaly_dataset(
            df,
            variable_of_interest,
            selected_variables
        )

        # Sauvegarde dans session_state
        st.session_state[
            "df_anomaly"
        ] = df_anomaly

        st.session_state[
            "anomaly_variables"
        ] = information["variables"]

        st.success(
            "✅ Jeu de données créé avec succès."
        )


# ==========================================================
# AFFICHAGE DU DATASET FINAL
# ==========================================================

if "df_anomaly" in st.session_state:

    df_anomaly = st.session_state[
        "df_anomaly"
    ]

    variables = st.session_state[
        "anomaly_variables"
    ]

    st.subheader(
        "📦 Jeu de données prêt pour la détection d'anomalies"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Lignes initiales",
            len(df)
        )

    with col2:
        st.metric(
            "Lignes après suppression des NA",
            len(df_anomaly)
        )

    with col3:
        st.metric(
            "Lignes supprimées",
            len(df) - len(df_anomaly)
        )

    with col4:
        st.metric(
            "Variables sélectionnées",
            len(variables)
        )


    st.write(
        "Variables utilisées :",
        ", ".join(variables)
    )


    st.dataframe(
        df_anomaly,
        use_container_width=True,
        height=500
    )


    # Vérification finale

    if df_anomaly.isna().sum().sum() == 0:

        st.success(
            "✅ Le jeu de données ne contient plus "
            "aucune valeur manquante."
        )

    else:

        st.error(
            "⚠️ Des valeurs manquantes sont encore présentes."
        )


    # Téléchargement CSV

    csv = df_anomaly.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Télécharger le jeu de données propre",
        data=csv,
        file_name="df_pour_detection_anomalies.csv",
        mime="text/csv"
    )
