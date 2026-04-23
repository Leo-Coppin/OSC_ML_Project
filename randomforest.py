import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"] 
folder_path = "DataShuffle"

rf_params = {
    'n_estimators': 500, 
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

print(f"🚀 Analyse des 7 shuffles - Détails par propriété\n")

for i in range(1, 8):
    try:
        X_train = pd.read_csv(os.path.join(folder_path, f"Data_Mordred_train_{i}.csv"), sep=";")
        X_test = pd.read_csv(os.path.join(folder_path, f"Data_Mordred_test_{i}.csv"), sep=";")
        y_train = pd.read_csv(os.path.join(folder_path, f"train_dataset_{i}.csv"), sep=";")
        y_test = pd.read_csv(os.path.join(folder_path, f"test_dataset_{i}.csv"), sep=";")

        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

        model = MultiOutputRegressor(RandomForestRegressor(**rf_params))
        model.fit(X_train, y_train[outputs])

        y_pred = model.predict(X_test)
        r2_raw = r2_score(y_test[outputs], y_pred, multioutput='raw_values')
        mae_raw = mean_absolute_error(y_test[outputs], y_pred, multioutput='raw_values')
        mse_raw = mean_squared_error(y_test[outputs], y_pred, multioutput='raw_values')

        df_shuffle = pd.DataFrame({
            'Propriété': outputs,
            'R²': r2_raw,
            'MAE': mae_raw,
            'MSE': mse_raw
        })

        print(f"📊 TABLEAU RÉSULTATS : SHUFFLE {i}")
        print("="*45)
        print(df_shuffle.round(4).to_string(index=False))
        print("-" * 45, "\n")

    except Exception as e:
        print(f"❌ Erreur sur le shuffle {i} : {e}")