
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Battery Anomaly Detection",
    page_icon="🔋",
    layout="wide"
)


# ============================================================
# PAGE D'ACCUEIL
# ============================================================

st.title("🔋 Battery Anomaly Detection")

st.markdown(
    """
    ## Bienvenue

    Cette application permet d'analyser les données de batteries
    et de détecter des anomalies à partir de différentes méthodes
    de Machine Learning.

    ### Fonctionnalités

    **📊 Données**
    
    Chargement et préparation des données du NASA Battery Dataset.

    **🔎 Détection d'anomalies**
    
    Détection des anomalies avec :
    - IQR
    - Isolation Forest univarié
    - Isolation Forest multivarié

    **📈 Visualisation**
    
    Analyse graphique des résultats et comparaison des méthodes.

    **📋 Rapport**
    
    Synthèse des résultats de l'analyse.
    """
)

