import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

# 1. Chargement des cibles de référence
y_train_all = pd.read_csv("train_dataset.csv", sep=';')
y_test_all = pd.read_csv("test_dataset.csv", sep=';')
outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]

datasets = [
    ("RDKit",   "Data_RDKit_train.csv",   "Data_RDKit_test.csv"),
    ("Mordred", "Data_Mordred_train.csv", "Data_Mordred_test.csv"),
    ("Morgan",  "Data_Morgan_train.csv",  "Data_Morgan_test.csv"),
    ("MACCS",   "Data_MACCS_train.csv",   "Data_MACCS_test.csv"),
    ("PubChem", "Data_PubChem_train.csv", "Data_PubChem_test.csv")
]

base_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputRegressor(base_rf)

print("🚀 Comparaison des descripteurs avec Random Forest :\n")

for name, train_f, test_f in datasets:
    try:
        # Chargement
        X_train = pd.read_csv(train_f, sep=";")
        X_test = pd.read_csv(test_f, sep=";")
        
        # --- SÉCURITÉ 1 : Alignement des colonnes ---
        all_cols = sorted(list(set(X_train.columns) | set(X_test.columns)))
        X_train = X_train.reindex(columns=all_cols, fill_value=0)
        X_test = X_test.reindex(columns=all_cols, fill_value=0)
        
        # --- SÉCURITÉ 2 : Nettoyage numérique ---
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

        # --- SÉCURITÉ 3 : Synchronisation avec Y ---
        # On ne garde que les indices qui existent dans X ET dans y
        common_train_idx = X_train.index.intersection(y_train_all.index)
        common_test_idx = X_test.index.intersection(y_test_all.index)
        
        X_train_final = X_train.loc[common_train_idx]
        y_train_final = y_train_all.loc[common_train_idx][outputs]
        
        X_test_final = X_test.loc[common_test_idx]
        y_test_final = y_test_all.loc[common_test_idx][outputs]

        # Entraînement
        model.fit(X_train_final, y_train_final)
        
        # Prédiction et Score
        y_pred = model.predict(X_test_final)
        r2 = r2_score(y_test_final, y_pred, multioutput='uniform_average')
        
        print(f"✅ {name:<8} | R² Moyen : {r2:.4f} | Samples : {len(X_test_final)}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ Erreur avec {name}: {e}")