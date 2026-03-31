import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib

datasets = [
    ("RDKit",   "Data_RDKit.csv",   "Output_RDKit.csv"),
    ("Mordred", "Data_Mordred.csv", "Output_Mordred.csv"),
    ("Morgan",  "Data_Morgan.csv",  "Output_Morgan.csv"),
    ("MACCS",   "Data_MACCS.csv",   "Output_MACCS.csv"),
    ("PubChem", "Data_PubChem.csv", "Output_Pubchem.csv")
]

target_columns = ['scaled_Voc', 'scaled_Jsc', 'scaled_FF', 'scaled_PCE',
                  'scaled_delta_LUMO', 'scaled_delta_HOMO']

# float32 safe max (~3e38), use a conservative threshold
FLOAT32_MAX = 3e38

results = []

for name, data_file, target_file in datasets:
    try:
        X = pd.read_csv(data_file, sep=';', low_memory=False)
        y_all = pd.read_csv(target_file, sep=';')

        # Force numeric, coerce errors → NaN
        X = X.apply(pd.to_numeric, errors='coerce')

        # Replace ±inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Replace values too large for float32 with NaN
        X = X.where(X.abs() <= FLOAT32_MAX, other=np.nan)

        # Drop columns with > 10% NaN
        X = X.dropna(axis=1, thresh=int(0.9 * len(X)))

        # Drop rows still containing NaN
        X = X.dropna()

        # Drop constant/zero-variance columns (useless features)
        X = X.loc[:, X.std() > 0]

        # Sync targets with surviving rows
        y = y_all.loc[X.index][target_columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(
            n_estimators=500, max_features='sqrt',
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({'Méthode': name, 'R2': r2, 'MAE': mae, 'Model': rf})
        print(f"{name:<15} | {r2:9.3f} | {mae:11.4f} | {X.shape[1]} features")

    except Exception as e:
        print(f"Error with {name}: {e}")

if results:
    best = max(results, key=lambda x: x['R2'])
    print("-" * 45)
    print(f"🏆 Best representation: {best['Méthode']} (R² = {best['R2']:.3f})")
    joblib.dump(best['Model'], f"best_multi_output_rf_{best['Méthode']}.pkl")