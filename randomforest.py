import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV # Nouveau pour l'optimisation

# 1. Charger le fichier de split maître
split_lookup = pd.read_csv("master_split.csv", sep=";", index_col=0)

# Liste des colonnes à exclure (tes "antisèches")
cols_to_exclude = [
    'scaled_HOMO_A', 'scaled_LUMO_A', 'scaled_EgCV_A', 'scaled_λ_A_absorption', 'scaled_EgA_opt',
    'scaled_HOMO_D', 'scaled_LUMO_D', 'scaled_EgCV_D', 'scaled_λ_D_absorption', 'scaled_EgD_opt'
]

target_cols = [
    'scaled_Voc', 'scaled_Jsc', 'scaled_FF', 'scaled_PCE', 
    'scaled_delta_LUMO', 'scaled_delta_HOMO'
]


datasets = [
    ("RDKit",   "Data_RDKit.csv",   "Output_RDKit.csv"),
    ("Mordred", "Data_Mordred.csv", "Output_Mordred.csv"),
    ("Morgan",  "Data_Morgan.csv",  "Output_Morgan.csv"),
    ("MACCS",   "Data_MACCS.csv",   "Output_MACCS.csv"),
    ("PubChem", "Data_PubChem.csv", "Output_Pubchem.csv")
]

for name, data_f, target_f in datasets:
    try:
        X_raw = pd.read_csv(data_f, sep=";")
        y_raw = pd.read_csv(target_f, sep=";")
        idx = X_raw.index.intersection(split_lookup.index)
        X = X_raw.loc[idx]
        X = X.drop(columns=[c for c in cols_to_exclude if c in X.columns])
        X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).clip(lower=-1e38, upper=1e38).fillna(0)
        X = X.loc[:, X.std() > 0] 
        y = y_raw.loc[idx][target_cols]
        sets = split_lookup.loc[idx]
        X_train, X_test = X[sets['set']=='train'], X[sets['set']=='test']
        y_train, y_test = y[sets['set']=='train'], y[sets['set']=='test']
        rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        r2 = r2_score(y_test, rf.predict(X_test))
        print(f"✅ {name:<10} | R² : {r2:.3f}")
    except Exception as e:
        print(f"❌ Erreur avec {name}: {e}")

'''
# =============================================================================
# 2. OPTIMISATION POUSSÉE DU MEILLEUR MODÈLE : MORDRED
# =============================================================================

print("\n🚀 Lancement de l'optimisation de Mordred...")

# Chargement spécifique de Mordred
X_mordred = pd.read_csv("Data_Mordred.csv", sep=";")
y_mordred = pd.read_csv("Output_Mordred.csv", sep=";")

# Nettoyage et Alignement (Identique à ton protocole rigoureux)
idx = X_mordred.index.intersection(split_lookup.index)
X = X_mordred.loc[idx].drop(columns=[c for c in cols_to_exclude if c in X_mordred.columns])
X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).clip(lower=-1e38, upper=1e38).fillna(0)
X = X.loc[:, X.std() > 0]
y = y_mordred.loc[idx][target_cols]
sets = split_lookup.loc[idx]

X_train, X_test = X[sets['set']=='train'], X[sets['set']=='test']
y_train, y_test = y[sets['set']=='train'], y[sets['set']=='test']

# Définition de la grille de recherche (Hyperparamètres)
param_distributions = {
    'n_estimators': [500, 1000, 1500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Création du moteur de recherche
# n_iter=10 signifie qu'on teste 10 combinaisons aléatoires parmis la grille
rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=10, 
    cv=3,       # Cross-validation à 3 plis
    verbose=2, 
    random_state=42, 
    n_jobs=-1,
    scoring='r2'
)

# Entraînement de l'optimisation
rf_random.fit(X_train, y_train)

# --- RÉSULTATS FINAUX ---
best_model = rf_random.best_estimator_
y_pred = best_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print(f"🏆 MEILLEUR SCORE MORDRED OPTIMISÉ : {final_r2:.4f}")
print(f"⚙️ Meilleurs paramètres : {rf_random.best_params_}")
print("="*40)

# Optionnel : Sauvegarde du meilleur modèle
# import joblib
# joblib.dump(best_model, "best_mordred_optimized.joblib")

# The optimized model gives a poorer score than the default one.
'''