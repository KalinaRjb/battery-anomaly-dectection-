

import os
import tempfile

import scipy.io as sio
import numpy as np
import pandas as pd


def safe_get(obj, field, default=np.nan):
    """
    Récupère un champ d'une structure MATLAB.
    Retourne default si le champ n'existe pas.
    """
    return getattr(obj, field, default)


def summarize_array(arr):
    """
    Retourne :
    mean, min, max, std

    Gère les arrays vides et les valeurs complexes.
    """
    arr = np.array(arr)

    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    return (
        arr.mean(),
        arr.min(),
        arr.max(),
        arr.std()
    )


def extract_battery_data(file_path, battery_name=None):

    if battery_name is None:
        battery_name = os.path.splitext(
            os.path.basename(file_path)
        )[0]

    mat = sio.loadmat(
        file_path,
        struct_as_record=False,
        squeeze_me=True
    )

    if battery_name not in mat:

        available_keys = [
            k for k in mat.keys()
            if not k.startswith("__")
        ]

        raise KeyError(
            f"Clé '{battery_name}' introuvable. "
            f"Clés disponibles : {available_keys}"
        )

    battery_struct = mat[battery_name]

    cycles = battery_struct.cycle

    # Si un seul cycle est présent
    if not isinstance(cycles, np.ndarray):
        cycles = np.array([cycles])

    rows = []

    for i, cyc in enumerate(cycles):

        cycle_type = safe_get(cyc, "type")

        # Conversion éventuelle de numpy.str_
        if isinstance(cycle_type, np.str_):
            cycle_type = str(cycle_type)

        ambient_temperature = safe_get(
            cyc,
            "ambient_temperature"
        )

        time = safe_get(cyc, "time")

        data = safe_get(cyc, "data")

        row = {
            "battery": battery_name,
            "cycle_index": i,
            "type": cycle_type,
            "ambient_temperature": ambient_temperature,
            "time": str(time)
        }

        # ==========================================================
        # CHARGE / DISCHARGE
        # ==========================================================

        if cycle_type in ("charge", "discharge"):

            # Voltage mesuré
            v_mean, v_min, v_max, v_std = summarize_array(
                safe_get(
                    data,
                    "Voltage_measured",
                    []
                )
            )

            # Courant mesuré
            i_mean, i_min, i_max, i_std = summarize_array(
                safe_get(
                    data,
                    "Current_measured",
                    []
                )
            )

            # Température mesurée
            t_mean, t_min, t_max, t_std = summarize_array(
                safe_get(
                    data,
                    "Temperature_measured",
                    []
                )
            )

            # Temps du cycle
            time_arr = np.array(
                safe_get(data, "Time", [])
            )

            row.update({
                "voltage_mean": v_mean,
                "voltage_min": v_min,
                "voltage_max": v_max,
                "voltage_std": v_std,

                "current_mean": i_mean,
                "current_min": i_min,
                "current_max": i_max,
                "current_std": i_std,

                "temperature_mean": t_mean,
                "temperature_min": t_min,
                "temperature_max": t_max,
                "temperature_std": t_std,

                "duration": (
                    time_arr.max()
                    if time_arr.size
                    else np.nan
                )
            })

            # ======================================================
            # CHARGE
            # ======================================================

            if cycle_type == "charge":

                cv_mean, _, _, _ = summarize_array(
                    safe_get(
                        data,
                        "Voltage_charge",
                        []
                    )
                )

                ci_mean, _, _, _ = summarize_array(
                    safe_get(
                        data,
                        "Current_charge",
                        []
                    )
                )

                row.update({
                    "voltage_charge_mean": cv_mean,
                    "current_charge_mean": ci_mean
                })

            # ======================================================
            # DISCHARGE
            # ======================================================

            if cycle_type == "discharge":

                # Pour le dataset NASA, les variables de décharge
                # sont Voltage_load et Current_load.

                lv_mean, _, _, _ = summarize_array(
                    safe_get(
                        data,
                        "Voltage_load",
                        []
                    )
                )

                li_mean, _, _, _ = summarize_array(
                    safe_get(
                        data,
                        "Current_load",
                        []
                    )
                )

                row.update({
                    "voltage_load_mean": lv_mean,
                    "current_load_mean": li_mean,
                    "capacity": safe_get(
                        data,
                        "Capacity"
                    )
                })

        # ==========================================================
        # IMPEDANCE
        # ==========================================================

        elif cycle_type == "impedance":

            sc_mean, _, _, _ = summarize_array(
                safe_get(
                    data,
                    "Sense_current",
                    []
                )
            )

            bc_mean, _, _, _ = summarize_array(
                safe_get(
                    data,
                    "Battery_current",
                    []
                )
            )

            cr_mean, _, _, _ = summarize_array(
                safe_get(
                    data,
                    "Current_ratio",
                    []
                )
            )

            bi_mean, _, _, _ = summarize_array(
                safe_get(
                    data,
                    "Battery_impedance",
                    []
                )
            )

            ri_mean, _, _, _ = summarize_array(
                safe_get(
                    data,
                    "Rectified_impedance",
                    []
                )
            )

            row.update({
                "sense_current_mean": sc_mean,
                "battery_current_mean": bc_mean,
                "current_ratio_mean": cr_mean,
                "battery_impedance_mean": bi_mean,
                "rectified_impedance_mean": ri_mean,
                "Re": safe_get(data, "Re"),
                "Rct": safe_get(data, "Rct")
            })

        rows.append(row)

    return pd.DataFrame(rows)


def process_uploaded_files(uploaded_files):

    all_data = []

    for uploaded_file in uploaded_files:

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_file.name

        try:

            # Nom réel du fichier uploadé
            battery_name = os.path.splitext(
                uploaded_file.name
            )[0]

            # Extraction
            df = extract_battery_data(
                temp_path,
                battery_name=battery_name
            )

            if df is not None and not df.empty:
                all_data.append(df)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_data:
        raise ValueError(
            "Aucune donnée n'a pu être extraite des fichiers."
        )

    return pd.concat(
        all_data,
        ignore_index=True
    )