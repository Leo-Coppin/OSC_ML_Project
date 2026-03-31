import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#defining best descriptors/ fingerprints
X_RDKit = pd.read_csv("Data_RDKit.csv", sep=';')
y_RDKit = pd.read_csv("Output_RDKit.csv", sep=';')

# Vérification valeurs trop grandes pour float32
float32_max = np.finfo(np.float32).max   # ~3.4e38
too_large = X_RDKit.columns[(X_RDKit.abs() > float32_max).any()].tolist()
X_RDKit = X_RDKit.clip(-float32_max, float32_max) 


X_Mordred = pd.read_csv("Data_Mordred.csv", sep=';', low_memory=False)
y_Mordred = pd.read_csv("Output_Mordred.csv", sep=';')

# verification if non numeric columns -> if non numeric columns non conversionable -> taken out of dataset
non_numeric = X_Mordred.select_dtypes(exclude=[np.number]).columns.tolist()
for col in non_numeric:
    X_Mordred[col] = pd.to_numeric(X_Mordred[col], errors='coerce')
nan_cols = X_Mordred.columns[X_Mordred.isna().any()].tolist()
X_Mordred = X_Mordred.drop(nan_cols, axis=1)
     
X_Morgan = pd.read_csv("Data_Morgan.csv", sep=';')
y_Morgan = pd.read_csv("Output_Morgan.csv", sep=';')

X_MACCS = pd.read_csv("Data_MACCS.csv", sep=';')
y_MACCS = pd.read_csv("Output_MACCS.csv", sep=';')

X_PubChem = pd.read_csv("Data_PubChem.csv", sep=';')
y_PubChem = pd.read_csv("Output_Pubchem.csv", sep=';')

Xy_pairs = [[X_RDKit, y_RDKit, "rdkit"], [X_Mordred, y_Mordred, "mordred"], [X_Morgan, y_Morgan, "morgan"], [X_MACCS, y_MACCS, "MACCS"], [X_PubChem, y_PubChem, "Pubchem"]]

outputs = ["scaled_Voc", "scaled_Jsc", "scaled_FF", "scaled_PCE", "scaled_delta_HOMO", "scaled_delta_LUMO"]

base_lr = LinearRegression()
multi_output_model = MultiOutputRegressor(base_lr) 
"""
for pairs in Xy_pairs : 
    X, y, name = pairs
    y = y[outputs]
                 
    #temporary -> waiting for leakless train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
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
"""

X = X_Morgan
y = y_Morgan[outputs]

#temporary -> waiting for leakless train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



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
