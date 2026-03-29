import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import SMILES_functions
import joblib
df_ci_final = pd.read_csv("temp_ci_score.csv", sep=';', index_col=0)
df_ci_final.index = df_ci_final.index.astype(int)

# Opening CSV File
df = pd.read_csv(
    "Data.csv",
    sep=";",
    dtype=str
)

# Choosen columns  : SMILES code, HOMO, LUMO, EgCV, λ absorption for Donor and Acceptor 
# In this dataset we’ll keep the columns Voc, Jsc, FF and PCE as outputs.
choosen_columns = ['SMILES_acc', 'SMILES_don', 'Voc', 'Jsc', 'FF', 'PCE', 'HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption', 'EgA_opt', 'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption', 'EgD_opt']
numerical_columns =  ['Voc', 'Jsc', 'FF', 'PCE', 'HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption', 'EgA_opt', 'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption', 'EgD_opt']

df_full = df[choosen_columns]

# without SMILES acceptor and SMILES donor -> only numerical columns
df_numerical = df[numerical_columns]

# Decimal point uniformisation : , -> .
df_numerical = df_numerical.replace(",", ".", regex=True)

#extract numbers 
for col in df_numerical.columns:
    df_numerical[col] = df_numerical[col].str.extract(r'([0-9.]+)')

df_numerical = df_numerical.apply(pd.to_numeric, errors="coerce")

# Keep only valid rows in numerical data (drop rows with NaN)
df_numerical = df_numerical.dropna()

# To keep fingerprint alignment: select the same valid indices in df_full
df_full = df_full.loc[df_numerical.index]


# spliting inputs and outputs from df_numerical
X_inputs = df_numerical[['HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption', 'EgA_opt',
                          'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption', 'EgD_opt']]

y_outputs = df_numerical[['Voc', 'Jsc', 'FF', 'PCE']]

y_outputs['delta_HOMO'] = X_inputs['HOMO_A'] - X_inputs['HOMO_D']
y_outputs['delta_LUMO']  = X_inputs['LUMO_A'] - X_inputs['LUMO_D']

# Scaling inputs
scaler_inputs = StandardScaler()
X_scaled = pd.DataFrame(
    scaler_inputs.fit_transform(X_inputs),
    columns=[f"scaled_{col}" for col in X_inputs.columns],
    index=df_numerical.index
)

# Scaling outputs
scaler_outputs = StandardScaler()
y_scaled = pd.DataFrame(
    scaler_outputs.fit_transform(y_outputs),
    columns=[f"scaled_{col}" for col in y_outputs.columns],
    index=df_numerical.index
)

joblib.dump(scaler_outputs, "scaler_outputs.pkl")
joblib.dump(scaler_inputs, "scaler_inputs.pkl")


X_scaled.to_csv("Data_Compatibility_score.csv", index=False, sep=';')
y_scaled.to_csv("Output_Compatibility_score.csv", index=False, sep=';')


# SMILES code processing
smiles_acceptor = df_full['SMILES_acc']
smiles_donor = df_full['SMILES_don']

# smiles_acceptor = smiles_acceptor.sample(n=30, random_state=42)
# smiles_donor = smiles_donor.sample(n=30, random_state=42)
# X_inputs = X_inputs.sample(n=30, random_state=42)
# y_outputs = y_outputs.sample(n=30, random_state=42)

lenght_columns = len(smiles_acceptor)

# --- RDKit Descriptors ---
print("RDKit Descriptors")
rdkit_list_acceptor = []
rdkit_list_donor = []
for smile in smiles_acceptor :
    rdkit_list_acceptor.append(SMILES_functions.get_rdkit_descriptors(smile))
for smile in smiles_donor : 
    rdkit_list_donor.append(SMILES_functions.get_rdkit_descriptors(smile))
    
df_rdkit_acceptor = pd.DataFrame(rdkit_list_acceptor, index=df_full.index)
df_rdkit_acceptor.columns=[f"rdkit_acceptor_{c}" for c in df_rdkit_acceptor.columns]
df_rdkit_donor = pd.DataFrame(rdkit_list_donor, index=df_full.index)
df_rdkit_donor.columns=[f"rdkit_donor_{c}" for c in df_rdkit_donor.columns]

df_rdkit = pd.concat([df_rdkit_acceptor, df_rdkit_donor, X_scaled], axis=1).dropna()

common_rdkit = df_rdkit.index.intersection(df_ci_final.index).intersection(y_scaled.index)
df_rdkit = df_rdkit.loc[common_rdkit]
df_rdkit_output = pd.concat([y_scaled.loc[common_rdkit], df_ci_final.loc[common_rdkit]], axis=1)

df_rdkit.to_csv("Data_RDKit.csv", index=False, sep=';')
df_rdkit_output.to_csv("Output_RDKit.csv", index=False, sep=";")


# --- Mordred Descriptors ---
print("Mordred Descriptors")
df_mordred_acc = SMILES_functions.get_mordred_descriptors(smiles_acceptor)
df_mordred_don = SMILES_functions.get_mordred_descriptors(smiles_donor)
df_mordred_acc.columns = [f"mordred_acceptor_{c}" for c in df_mordred_acc.columns]
df_mordred_don.columns = [f"mordred_donor_{c}" for c in df_mordred_don.columns]

df_mordred = pd.concat([df_mordred_acc, df_mordred_don, X_scaled], axis=1).dropna()

common_mordred = df_mordred.index.intersection(df_ci_final.index).intersection(y_scaled.index)
df_mordred = df_mordred.loc[common_mordred]
df_mordred_output = pd.concat([y_scaled.loc[common_mordred], df_ci_final.loc[common_mordred]], axis=1)

df_mordred.to_csv("Data_Mordred.csv", index=False, sep=';')
df_mordred_output.to_csv("Output_Mordred.csv", index=False, sep=";")


# --- Morgan Fingerprints ---
print("Morgan Fingerprints")
morgan_matrix_acc = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_acceptor])
df_morgan_acc = pd.DataFrame(morgan_matrix_acc, index=df_full.index, columns=[f"morgan_acc_{i}" for i in range(morgan_matrix_acc.shape[1])])
morgan_matrix_don = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_donor])
df_morgan_don = pd.DataFrame(morgan_matrix_don, index=df_full.index, columns=[f"morgan_don_{i}" for i in range(morgan_matrix_don.shape[1])])

df_morgan = pd.concat([df_morgan_acc, df_morgan_don, X_scaled], axis=1).dropna()

common_morgan = df_morgan.index.intersection(df_ci_final.index).intersection(y_scaled.index)
df_morgan = df_morgan.loc[common_morgan]
df_morgan_output = pd.concat([y_scaled.loc[common_morgan], df_ci_final.loc[common_morgan]], axis=1)

df_morgan.to_csv("Data_Morgan.csv", index=False, sep=';')
df_morgan_output.to_csv("Output_Morgan.csv", index=False, sep=";")


# --- MACCS Keys fingerprints ---
print("MACCS Keys Fingerprints")
maccs_matrix_acc = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_acceptor])
df_maccs_acc = pd.DataFrame(maccs_matrix_acc, index=df_full.index, columns=[f"maccs_acc_{i}" for i in range(maccs_matrix_acc.shape[1])])
maccs_matrix_don = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_donor])
df_maccs_don = pd.DataFrame(maccs_matrix_don, index=df_full.index, columns=[f"maccs_don_{i}" for i in range(maccs_matrix_don.shape[1])])

df_maccs = pd.concat([df_maccs_acc, df_maccs_don, X_scaled], axis=1).dropna()

common_maccs = df_maccs.index.intersection(df_ci_final.index).intersection(y_scaled.index)
df_maccs = df_maccs.loc[common_maccs]
df_maccs_output = pd.concat([y_scaled.loc[common_maccs], df_ci_final.loc[common_maccs]], axis=1)

df_maccs.to_csv("Data_MACCS.csv", index=False, sep=';')
df_maccs_output.to_csv("Output_MACCS.csv", index=False, sep=";")


# --- PubChem Fingerprints ---
print("PubChem Fingerprints")
pc_matrix_acc = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_acceptor])
df_pc_acc = pd.DataFrame(pc_matrix_acc, index=df_full.index, columns=[f"pubchem_acc_{i}" for i in range(pc_matrix_acc.shape[1])])
pc_matrix_don = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_donor])
df_pc_don = pd.DataFrame(pc_matrix_don, index=df_full.index, columns=[f"pubchem_don_{i}" for i in range(pc_matrix_don.shape[1])])

df_pubchem = pd.concat([df_pc_acc, df_pc_don, X_scaled], axis=1).dropna()

common_pubchem = df_pubchem.index.intersection(df_ci_final.index).intersection(y_scaled.index)
df_pubchem = df_pubchem.loc[common_pubchem]
df_pubchem_output = pd.concat([y_scaled.loc[common_pubchem], df_ci_final.loc[common_pubchem]], axis=1)

df_pubchem.to_csv("Data_PubChem.csv", index=False, sep=';')
df_pubchem_output.to_csv("Output_PubChem.csv", index=False, sep=";")

print("Data Preparation")