import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_train = pd.read_csv("train_dataset.csv", sep=';')
y_test = pd.read_csv("test_dataset.csv", sep=';')

outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]
y_train = y_train[outputs]
y_test = y_test[outputs]

#defining best descriptors/ fingerprints
X_train_RDKit = pd.read_csv("Data_RDKit_train.csv", sep=';')
X_test_RDKit = pd.read_csv("Data_RDKit_test.csv", sep=';')

# Vérification valeurs trop grandes pour float32
float32_max = np.finfo(np.float32).max   # ~3.4e38
too_large = X_train_RDKit.columns[(X_train_RDKit.abs() > float32_max).any()].tolist()
X_train_RDKit = X_train_RDKit.clip(-float32_max, float32_max) 

X_train_Mordred = pd.read_csv("Data_Mordred_train.csv", sep=';', low_memory=False)
X_test_Mordred = pd.read_csv("Data_Mordred_test.csv", sep=';')

# verification if non numeric columns -> if non numeric columns non conversionable -> taken out of dataset
non_numeric = X_train_Mordred.select_dtypes(exclude=[np.number]).columns.tolist()
for col in non_numeric:
    X_train_Mordred[col] = pd.to_numeric(X_train_Mordred[col], errors='coerce')
nan_cols = X_train_Mordred.columns[X_train_Mordred.isna().any()].tolist()
X_train_Mordred = X_train_Mordred.drop(nan_cols, axis=1)
    
non_numeric = X_test_Mordred.select_dtypes(exclude=[np.number]).columns.tolist()
for col in non_numeric:
    X_test_Mordred[col] = pd.to_numeric(X_test_Mordred[col], errors='coerce')
nan_cols = X_test_Mordred.columns[X_test_Mordred.isna().any()].tolist()
X_test_Mordred = X_test_Mordred.drop(nan_cols, axis=1)

all_features = sorted(list(set(X_train_Mordred.columns) | set(X_test_Mordred.columns)))

# 2. On aligne X_train sur cette liste globale
X_train_Mordred = X_train_Mordred.reindex(columns=all_features, fill_value=0)
# 3. On aligne X_test sur cette MÊME liste globale
X_test_Mordred = X_test_Mordred.reindex(columns=all_features, fill_value=0)
     
X_train_Morgan = pd.read_csv("Data_Morgan_train.csv", sep=';')
X_test_Morgan = pd.read_csv("Data_Morgan_test.csv", sep=';')

X_train_MACCS = pd.read_csv("Data_MACCS_train.csv", sep=';')
X_test_MACCS = pd.read_csv("Data_MACCS_test.csv", sep=';')

X_train_Pubchem = pd.read_csv("Data_PubChem_train.csv", sep=';')
X_test_Pubchem = pd.read_csv("Data_PubChem_test.csv", sep=';')

train_test_pairs = [[X_train_RDKit, X_test_RDKit, "rdkit"], [X_train_Mordred, X_test_Mordred, "mordred"], [X_train_Morgan, X_test_Morgan, "morgan"], [X_train_MACCS, X_test_MACCS, "MACCS"], [X_train_Pubchem, X_test_Pubchem, "Pubchem"]]


base_lr = LinearRegression()
multi_output_model = MultiOutputRegressor(base_lr) 
"""
for pairs in train_test_pairs : 
    X_train, X_test, name = pairs
                 
    #temporary -> waiting for leakless train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    
    multi_output_model.fit(X_train, y_train)
    y_pred = multi_output_model.predict(X_test)
    
    r2   = r2_score(y_test, y_pred, multioutput='raw_values')
    mae  = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
    
    results_df = pd.DataFrame({
        'R2'  : r2,
        'MAE' : mae,
        'RMSE': rmse
    }, index=outputs)
    
    print(f"Results for {name} :")
    print(results_df.round(4))
#"""
X_train = X_train_Pubchem
X_test = X_test_Pubchem


model = MultiOutputRegressor(base_lr) 

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

results = pd.DataFrame({
    'R2'  : r2,
    'MAE' : mae,
    'RMSE': rmse
}, index=outputs)



print("Results for Voc, Jsc FF, PCE, delta HOMO and delta LUMO:")
print(results.round(4))
