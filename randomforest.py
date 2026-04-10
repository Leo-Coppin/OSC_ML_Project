'''import pandas as pd
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
        print(f"❌ Erreur avec {name}: {e}")'''

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

# 1. Chargement des cibles
y_train_all = pd.read_csv("train_dataset.csv", sep=';')
y_test_all = pd.read_csv("test_dataset.csv", sep=';')
outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]

# 2. Choix du meilleur dataset (Mordred ou RDKit)
NAME = "Mordred"
X_train_path = "Data_Mordred_train.csv"
X_test_path = "Data_Mordred_test.csv"

print(f"🛠️ Optimisation du modèle : {NAME}")

# Chargement et nettoyage rigoureux
X_train = pd.read_csv(X_train_path, sep=";").apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = pd.read_csv(X_test_path, sep=";").apply(pd.to_numeric, errors='coerce').fillna(0)

# Synchronisation stricte des index
common_train_idx = X_train.index.intersection(y_train_all.index)
common_test_idx = X_test.index.intersection(y_test_all.index)

X_train_final = X_train.loc[common_train_idx]
y_train_final = y_train_all.loc[common_train_idx][outputs]
X_test_final = X_test.loc[common_test_idx]
y_test_final = y_test_all.loc[common_test_idx][outputs]

# 3. Définition de la grille d'optimisation
# On cherche les meilleurs paramètres pour l'estimateur de base
param_grid = {
    'estimator__n_estimators': [500], 
    'estimator__max_depth': [15, 20, None],
    'estimator__max_features': ['sqrt', 'log2'], 
    'estimator__min_samples_leaf': [1, 2]
}

# Modèle de base
base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
multi_model = MultiOutputRegressor(base_rf)

# 4. Recherche par Validation Croisée (GridSearch)
print("⏳ Recherche de la meilleure configuration (cela peut prendre du temps)...")
grid_search = GridSearchCV(
    multi_model, 
    param_grid, 
    cv=3, 
    scoring='r2', 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train_final, y_train_final)

# 1. Récupération du meilleur modèle
best_model = grid_search.best_estimator_

cols_at_fit = X_train_final.columns
X_test_final = X_test_final.reindex(columns=cols_at_fit, fill_value=0)

y_pred = best_model.predict(X_test_final)

# 2. Calcul des métriques globales
r2_global = r2_score(y_test_final, y_pred, multioutput='uniform_average')
mae_global = mean_absolute_error(y_test_final, y_pred)
mse_global = mean_squared_error(y_test_final, y_pred)

print("\n" + "="*50)
print(f"🏆 RÉSULTATS GLOBAUX DU MODÈLE {NAME.upper()} OPTIMISÉ")
print("="*50)
print(f"✨ R² Moyen  : {r2_global:.4f}")
print(f"📉 MAE Moyenne : {mae_global:.4f}")
print(f"🧪 MSE Moyenne : {mse_global:.4f}")
print(f"⚙️ Paramètres retenus : {grid_search.best_params_}")
print("-" * 50)

# 3. Détail par propriété
r2_per_col = r2_score(y_test_final, y_pred, multioutput='raw_values')
mae_per_col = mean_absolute_error(y_test_final, y_pred, multioutput='raw_values')
mse_per_col = mean_squared_error(y_test_final, y_pred, multioutput='raw_values')

results_detailed = pd.DataFrame({
    'R²': r2_per_col,
    'MAE': mae_per_col,
    'MSE': mse_per_col
}, index=outputs)

print("\n📊 DÉTAILS PAR PROPRIÉTÉ :")
print(results_detailed.round(4))
print("="*50)

# 4. Sauvegarde
joblib.dump(best_model, f"best_rf_{NAME.lower()}_optimized.pkl")