
import streamlit as st
import pandas as pd

from utils.data_processing import process_uploaded_files


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Données - Battery Anomaly Detection",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITRE
# ============================================================

st.title("📊 Données batteries")

st.markdown(
    """
    Cette page permet de charger les fichiers `.mat` du
    NASA Battery Dataset et d'extraire les caractéristiques
    de chaque cycle de batterie.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Chargement des données")

uploaded_files = st.sidebar.file_uploader(
    "Charger les fichiers MATLAB (.mat)",
    type=["mat"],
    accept_multiple_files=True
)


# ============================================================
# MESSAGE SI AUCUN FICHIER
# ============================================================

if not uploaded_files:

    st.info(
        "👈 Chargez un ou plusieurs fichiers `.mat` "
        "depuis la barre latérale."
    )

    st.markdown(
        """
        ### Fichiers recommandés

        Vous pouvez charger par exemple :

        - `B0005.mat`
        - `B0006.mat`
        - `B0007.mat`
        - `B0018.mat`
        """
    )

    # Si des données existent déjà en mémoire,
    # on ne les efface pas.
    if "df_all" not in st.session_state:
        st.stop()


# ============================================================
# TRAITEMENT DES FICHIERS
# ============================================================

if uploaded_files:

    with st.spinner("Traitement des fichiers..."):

        try:

            df_all = process_uploaded_files(
                uploaded_files
            )

            # Stockage dans la session Streamlit
            st.session_state["df_all"] = df_all

        except Exception as e:

            st.error(
                f"❌ Une erreur est survenue lors du traitement : {e}"
            )

            st.stop()

else:

    # Récupération des données déjà chargées
    df_all = st.session_state["df_all"]


# ============================================================
# MESSAGE DE SUCCÈS
# ============================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} fichier(s) traité(s) avec succès."
    )

else:

    st.info(
        "📂 Données précédemment chargées disponibles."
    )


# ============================================================
# INDICATEURS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Batteries",
        df_all["battery"].nunique()
    )

with col2:

    st.metric(
        "Cycles",
        len(df_all)
    )

with col3:

    st.metric(
        "Variables",
        len(df_all.columns)
    )

with col4:

    st.metric(
        "Types de cycles",
        df_all["type"].nunique()
    )


# ============================================================
# ONGLETS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Données",
        "📋 Description des variables",
        "📈 Statistiques"
    ]
)


# ============================================================
# ONGLET 1 : DONNÉES
# ============================================================

with tab1:

    st.subheader("Données extraites")

    st.dataframe(
        df_all,
        use_container_width=True,
        height=500
    )

    # --------------------------------------------------------
    # Téléchargement CSV
    # --------------------------------------------------------

    csv = df_all.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Télécharger les données CSV",
        data=csv,
        file_name="df_all.csv",
        mime="text/csv"
    )


# ============================================================
# ONGLET 2 : DESCRIPTION DES VARIABLES
# ============================================================

with tab2:

    st.subheader(
        "Description des variables"
    )

    description = pd.DataFrame({

        "Variable": df_all.columns,

        "Type": [
            str(df_all[col].dtype)
            for col in df_all.columns
        ],

        "Valeurs manquantes": [
            df_all[col].isna().sum()
            for col in df_all.columns
        ],

        "% manquant": [
            round(
                df_all[col].isna().mean() * 100,
                2
            )
            for col in df_all.columns
        ],

        "Valeurs uniques": [
            df_all[col].nunique()
            for col in df_all.columns
        ]
    })

    st.dataframe(
        description,
        use_container_width=True
    )


# ============================================================
# ONGLET 3 : STATISTIQUES
# ============================================================

with tab3:

    st.subheader(
        "Statistiques descriptives"
    )

    st.dataframe(
        df_all.describe(
            include="all"
        ).T,
        use_container_width=True
    )


# ============================================================
# ANALYSE PAR BATTERIE
# ============================================================

st.divider()

st.subheader("🔋 Analyse par batterie")

selected_battery = st.selectbox(
    "Sélectionner une batterie",
    sorted(df_all["battery"].unique())
)

battery_df = df_all[
    df_all["battery"] == selected_battery
]

st.write(
    f"Nombre de cycles : **{len(battery_df)}**"
)

st.dataframe(
    battery_df,
    use_container_width=True
)


# ============================================================
# RÉPARTITION DES CYCLES
# ============================================================

st.subheader("🔄 Répartition des cycles")

cycle_counts = (
    df_all["type"]
    .value_counts()
)

st.bar_chart(
    cycle_counts
)


