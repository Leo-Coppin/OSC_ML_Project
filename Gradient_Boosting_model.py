from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Y_train = pd.read_csv("train_dataset.csv", sep=';')
Y_test = pd.read_csv("test_dataset.csv", sep=';')

outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]
Y_train = Y_train[outputs]
Y_test = Y_test[outputs]

X_train_RDKit = pd.read_csv("Data_RDKit_train.csv", sep=';')
X_test_RDKit = pd.read_csv("Data_RDKit_test.csv", sep=';')

X_train_Mordred = pd.read_csv("Data_Mordred_train.csv", sep=';', low_memory=False)
X_test_Mordred = pd.read_csv("Data_Mordred_test.csv", sep=';')

X_train_Morgan = pd.read_csv("Data_Morgan_train.csv", sep=';')
X_test_Morgan = pd.read_csv("Data_Morgan_test.csv", sep=';')

X_train_MACCS = pd.read_csv("Data_MACCS_train.csv", sep=';')
X_test_MACCS = pd.read_csv("Data_MACCS_test.csv", sep=';')

X_train_Pubchem = pd.read_csv("Data_PubChem_train.csv", sep=';')
X_test_Pubchem = pd.read_csv("Data_PubChem_test.csv", sep=';')


def preprocess_for_xgboost(X_train, X_test):
    # Convert everything possible to numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Replace inf values with NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Clip values that are too large for float32
    float32_max = np.finfo(np.float32).max
    X_train = X_train.clip(-float32_max, float32_max)
    X_test = X_test.clip(-float32_max, float32_max)

    # Keep only columns that are not entirely NaN in train
    valid_cols = X_train.columns[~X_train.isna().all()]
    X_train = X_train[valid_cols]
    X_test = X_test.reindex(columns=valid_cols)

    # Align train and test columns
    all_features = sorted(list(set(X_train.columns) | set(X_test.columns)))
    X_train = X_train.reindex(columns=all_features, fill_value=np.nan)
    X_test = X_test.reindex(columns=all_features, fill_value=np.nan)

    # Drop columns still completely NaN in both train and test
    cols_to_keep = ~(X_train.isna().all() & X_test.isna().all())
    X_train = X_train.loc[:, cols_to_keep]
    X_test = X_test.loc[:, cols_to_keep]

    # Final safety: make sure all values are float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, X_test


X_train_RDKit, X_test_RDKit = preprocess_for_xgboost(X_train_RDKit, X_test_RDKit)
X_train_Mordred, X_test_Mordred = preprocess_for_xgboost(X_train_Mordred, X_test_Mordred)
X_train_Morgan, X_test_Morgan = preprocess_for_xgboost(X_train_Morgan, X_test_Morgan)
X_train_MACCS, X_test_MACCS = preprocess_for_xgboost(X_train_MACCS, X_test_MACCS)
X_train_Pubchem, X_test_Pubchem = preprocess_for_xgboost(X_train_Pubchem, X_test_Pubchem)

train_test_pairs = [
    [X_train_RDKit, X_test_RDKit, "rdkit"],
    [X_train_Mordred, X_test_Mordred, "mordred"],
    [X_train_Morgan, X_test_Morgan, "morgan"],
    [X_train_MACCS, X_test_MACCS, "MACCS"],
    [X_train_Pubchem, X_test_Pubchem, "Pubchem"]
]
def gradient_boosting(
    train_x,
    train_y,
    test_x,
    test_y,
    target_cols=['Voc', 'Jsc', 'FF', 'PCE', 'delta_HOMO', 'delta_LUMO']
):
    # Make sure y uses only the expected target columns
    train_y = train_y[target_cols].copy()
    test_y = test_y[target_cols].copy()

    model = XGBRegressor(
        objective="reg:squarederror",
        multi_strategy="one_output_per_tree",
        n_estimators=700,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=8,
        random_state=42
    )

    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    pred_df = pd.DataFrame(
        y_pred,
        columns=target_cols,
        index=test_y.index
    )

    results = []

    for col in target_cols:
        mae = mean_absolute_error(test_y[col], pred_df[col])
        mse = mean_squared_error(test_y[col], pred_df[col])
        r2 = r2_score(test_y[col], pred_df[col])

        results.append({
            "Target": col,
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        })

        print(f"{col}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2:   {r2:.4f}")
        print()

    results_df = pd.DataFrame(results)

    return model, pred_df, results_df

all_results = {}

for X_train, X_test, name in train_test_pairs:
    print(f"\n===== {name} =====")
    model, pred_df, results_df = gradient_boosting(X_train, Y_train, X_test, Y_test)
    all_results[name] = {
        "model": model,
        "predictions": pred_df,
        "results": results_df
    }