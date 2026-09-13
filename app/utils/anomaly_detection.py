
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


RANDOM_STATE = 42


# ==========================================================
# IQR
# ==========================================================

def detect_anomalies_iqr(
    df,
    variable,
    k=1.5
):
    """
    Détection d'anomalies avec la méthode IQR.

    Une observation est considérée comme anormale si :

    valeur < Q1 - k * IQR

    ou

    valeur > Q3 + k * IQR

    k = 1.5 : convention classique
    k = 3   : détection plus restrictive
    """

    series = df[variable]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    is_anomaly = (
        (series < lower_bound) |
        (series > upper_bound)
    ).astype(int)

    distance = np.where(
        series < lower_bound,
        lower_bound - series,
        np.where(
            series > upper_bound,
            series - upper_bound,
            0
        )
    )

    if iqr != 0:
        iqr_score = distance / iqr
    else:
        iqr_score = distance

    result = df.copy()

    result["iqr_score"] = iqr_score

    result["anomaly_iqr"] = is_anomaly

    information = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

    return result, information


# ==========================================================
# ISOLATION FOREST
# ==========================================================

def detect_anomalies_isolation_forest(
    df,
    feature_cols,
    contamination=0.05
):
    """
    Détection d'anomalies avec Isolation Forest.

    feature_cols :
        - une variable -> Isolation Forest univarié
        - plusieurs variables -> Isolation Forest multivarié

    contamination :
        proportion attendue d'anomalies.
    """

    X = df[feature_cols].copy()

    # Standardisation
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_scaled)

    # Score brut :
    # plus élevé = observation normale
    raw_scores = model.decision_function(X_scaled)

    # Prédiction :
    # 1 = normal
    # -1 = anomalie
    predictions = model.predict(X_scaled)

    # Transformation en score :
    # 0 = peu anormal
    # 1 = très anormal

    score_range = (
        raw_scores.max() -
        raw_scores.min()
    )

    if score_range != 0:

        anomaly_score = (
            raw_scores.max() -
            raw_scores
        ) / score_range

    else:

        anomaly_score = np.zeros(
            len(raw_scores)
        )

    result = df.copy()

    result["isoforest_score"] = anomaly_score

    result["anomaly_isoforest"] = (
        predictions == -1
    ).astype(int)

    return result, model


# ==========================================================
# TABLEAU DE COMPARAISON
# ==========================================================

def build_comparison_table(
    result_iqr,
    result_uni,
    result_multi,
    variable
):
    """
    Construit un tableau regroupant les résultats
    des trois méthodes.
    """

    comparison = pd.DataFrame(
        index=result_iqr.index
    )

    comparison[variable] = (
        result_iqr[variable]
    )

    comparison["iqr_score"] = (
        result_iqr["iqr_score"]
    )

    comparison["anomaly_iqr"] = (
        result_iqr["anomaly_iqr"]
    )

    comparison["isoforest_uni_score"] = (
        result_uni["isoforest_score"]
    )

    comparison["anomaly_isoforest_uni"] = (
        result_uni["anomaly_isoforest"]
    )

    comparison["isoforest_multi_score"] = (
        result_multi["isoforest_score"]
    )

    comparison["anomaly_isoforest_multi"] = (
        result_multi["anomaly_isoforest"]
    )

    anomaly_columns = [
        "anomaly_iqr",
        "anomaly_isoforest_uni",
        "anomaly_isoforest_multi"
    ]

    comparison[anomaly_columns] = (
        comparison[anomaly_columns]
        .fillna(0)
        .astype(int)
    )

    comparison["n_methods_flagged"] = (
        comparison[anomaly_columns]
        .sum(axis=1)
    )

    return comparison


# ==========================================================
# PCA POUR VISUALISER LE MULTIVARIE
# ==========================================================

def create_pca_projection(
    result_multi,
    feature_cols
):
    """
    Projection des données multivariées en 2 dimensions
    avec PCA.
    """

    X = result_multi[feature_cols]

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    coordinates = pca.fit_transform(X)

    result_plot = result_multi.copy()

    result_plot["PCA1"] = coordinates[:, 0]
    result_plot["PCA2"] = coordinates[:, 1]

    explained_variance = (
        pca.explained_variance_ratio_
    )

    return result_plot, explained_variance
