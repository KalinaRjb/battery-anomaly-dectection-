
import scipy.io
import scipy.io as sio
import glob
import os
import numpy as np
import pandas as pd


data_dir = r"C:\Users\kalin\OneDrive\Documents\SAFT\Data\archive\Battery_DataSet\Battery_DataSet"

# Charger tous les fichiers .mat du dossier
mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
for f in mat_files:
    print(f)

# %%
# Dictionnaire pour stocker les résultats de chaque batterie
batteries = {}

for idx, file_path in enumerate(mat_files):
    battery_name = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        
        if battery_name not in mat:
            print(f"⚠️  {battery_name} : clé introuvable dans le fichier — clés disponibles : "
                  f"{[k for k in mat.keys() if not k.startswith('__')]}")
            continue
        
        battery_struct = mat[battery_name]
        cycles = battery_struct.cycle
        
        # Stocke pour réutilisation ultérieure
        batteries[battery_name] = cycles
        
        print(f"✅ {battery_name} : {len(cycles)} cycles trouvés "
              f"(1er cycle : {cycles[0].type})")
        
    except Exception as e:
        print(f"❌ {battery_name} : erreur — {e}")

print(f"\n{len(batteries)} batteries chargées avec succès sur {len(mat_files)} fichiers.")

# %%
def safe_get(obj, field, default=np.nan):
    """Récupère un champ d'une structure MATLAB s'il existe, sinon retourne default."""
    return getattr(obj, field, default)

# %%
def summarize_array(arr):
    """Retourne (mean, min, max, std) d'un array numpy, en gérant les valeurs vides/complexes."""
    arr = np.array(arr)
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if np.iscomplexobj(arr):
        arr = np.abs(arr)  # magnitude pour les grandeurs complexes (impedance)
    return arr.mean(), arr.min(), arr.max(), arr.std()

# %%
def extract_battery_dataframe(file_path):
    """Transforme un fichier .mat (format NASA battery dataset) en DataFrame, une ligne par cycle."""
    
    battery_name = os.path.splitext(os.path.basename(file_path))[0]
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    if battery_name not in mat:
        raise KeyError(f"Clé '{battery_name}' introuvable. Clés disponibles : "
                        f"{[k for k in mat.keys() if not k.startswith('__')]}")
    
    battery_struct = mat[battery_name]
    cycles = battery_struct.cycle
    
    rows = []
    
    for i, cyc in enumerate(cycles):
        cycle_type = cyc.type
        ambient_temperature = safe_get(cyc, "ambient_temperature")
        time = safe_get(cyc, "time")
        data = cyc.data
        
        row = {
            "battery": battery_name,
            "cycle_index": i,
            "type": cycle_type,
            "ambient_temperature": ambient_temperature,
            "time": str(time),  # vecteur date MATLAB -> stocké en string pour simplicité
        }
        
        if cycle_type in ("charge", "discharge"):
            v_mean, v_min, v_max, v_std = summarize_array(safe_get(data, "Voltage_measured", []))
            i_mean, i_min, i_max, i_std = summarize_array(safe_get(data, "Current_measured", []))
            t_mean, t_min, t_max, t_std = summarize_array(safe_get(data, "Temperature_measured", []))
            time_arr = np.array(safe_get(data, "Time", []))
            
            row.update({
                "voltage_mean": v_mean, "voltage_min": v_min, "voltage_max": v_max, "voltage_std": v_std,
                "current_mean": i_mean, "current_min": i_min, "current_max": i_max, "current_std": i_std,
                "temperature_mean": t_mean, "temperature_min": t_min, "temperature_max": t_max, "temperature_std": t_std,
                "duration": time_arr.max() if time_arr.size else np.nan,
            })
            
            if cycle_type == "charge":
                cv_mean, cv_min, cv_max, cv_std = summarize_array(safe_get(data, "Voltage_charge", []))
                ci_mean, ci_min, ci_max, ci_std = summarize_array(safe_get(data, "Current_charge", []))
                row.update({
                    "voltage_charge_mean": cv_mean,
                    "current_charge_mean": ci_mean,
                })
            
            if cycle_type == "discharge":
                lv_mean, lv_min, lv_max, lv_std = summarize_array(safe_get(data, "Voltage_charge", []))  # tension mesurée à la charge (load)
                li_mean, li_min, li_max, li_std = summarize_array(safe_get(data, "Current_charge", []))  # courant mesuré à la charge (load)
                row.update({
                    "voltage_load_mean": lv_mean,
                    "current_load_mean": li_mean,
                    "capacity": safe_get(data, "Capacity"),
                })
        
        elif cycle_type == "impedance":
            sc_mean, sc_min, sc_max, sc_std = summarize_array(safe_get(data, "Sense_current", []))
            bc_mean, bc_min, bc_max, bc_std = summarize_array(safe_get(data, "Battery_current", []))
            cr_mean, cr_min, cr_max, cr_std = summarize_array(safe_get(data, "Current_ratio", []))
            bi_mean, bi_min, bi_max, bi_std = summarize_array(safe_get(data, "Battery_impedance", []))
            ri_mean, ri_min, ri_max, ri_std = summarize_array(safe_get(data, "Rectified_impedance", []))
            
            row.update({
                "sense_current_mean": sc_mean,
                "battery_current_mean": bc_mean,
                "current_ratio_mean": cr_mean,
                "battery_impedance_mean": bi_mean,   # magnitude (valeurs complexes -> abs)
                "rectified_impedance_mean": ri_mean, # magnitude
                "Re": safe_get(data, "Re"),
                "Rct": safe_get(data, "Rct"),
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)

# ======================================================================
# Application à tous les fichiers du README (B0005, B0006, B0007, B0018)
# ======================================================================

# %%
battery_files = {
    "B0005": "B0005.mat",
    "B0006": "B0006.mat",
    "B0007": "B0007.mat",
    "B0018": "B0018.mat",
}

# adapte le chemin si tes fichiers sont dans un dossier
base_path = r"C:\Users\kalin\OneDrive\Documents\SAFT\Data\archive\Battery_DataSet\Battery_DataSet"  

dfs = {}
for name, filename in battery_files.items():
    file_path = os.path.join(base_path, filename)
    try:
        dfs[name] = extract_battery_dataframe(file_path)
        print(f"✅ {name} : {len(dfs[name])} cycles extraits")
    except Exception as e:
        print(f"❌ {name} : erreur — {e}")

# Concatène tout dans un seul DataFrame global
df_all = pd.concat(dfs.values(), ignore_index=True)
print(df_all.shape)
df_all.head()

# %%


# %%
df_all.dtypes

# %%
df_all['time']

# %%
dossier = r"C:\Users\kalin\OneDrive\Documents\SAFT\Data"

df_all.to_csv(dossier + r"\df_all.csv", index=False)


