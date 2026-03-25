import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import SMILES_functions
import joblib


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

X_inputs['delta_LUMO']    = X_inputs['LUMO_A'] - X_inputs['LUMO_D']
X_inputs['delta_HOMO']    = X_inputs['HOMO_A'] - X_inputs['HOMO_D']
X_inputs['HOMO_D_LUMO_A'] = X_inputs['HOMO_D'] - X_inputs['LUMO_A']  # gap interfacial → lié à Voc
X_inputs['delta_Eg']      = X_inputs['EgCV_A'] - X_inputs['EgCV_D']
X_inputs['delta_lambda']  = X_inputs['λ_A_absorption'] - X_inputs['λ_D_absorption']

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
"""
#RDKit descriptors
print("RDKit Descriptors")
rdkit_list_acceptor = []
rdkit_list_donor = []
for smile in smiles_acceptor :
    rdkit_list_acceptor.append(SMILES_functions.get_rdkit_descriptors(smile))

for smile in smiles_donor : 
    rdkit_list_donor.append(SMILES_functions.get_rdkit_descriptors(smile))
    
df_rdkit_acceptor = pd.DataFrame(rdkit_list_acceptor, index = df_full.index)
df_rdkit_acceptor.columns=[f"rdkit_acceptor_{c}" for c in df_rdkit_acceptor.columns]

df_rdkit_donor = pd.DataFrame(rdkit_list_donor, index = df_full.index)
df_rdkit_donor.columns=[f"rdkit_donor_{c}" for c in df_rdkit_donor.columns]

# concatenantion Smiles acceptor, donor and input numerical Data
df_rdkit = pd.concat([df_rdkit_acceptor, df_rdkit_donor, X_scaled], axis=1)

# Dropping uncomplete lines
df_rdkit = df_rdkit.dropna()
df_rdkit_output = y_scaled.loc[df_rdkit.index]

# Save in csv file
df_rdkit.to_csv("Data_RDKit.csv", index=False, sep=';')
df_rdkit_output.to_csv("Output_RDKit.csv", index=False, sep=";")


# Mordred Descriptors
print("Mordred Descriptors")
df_mordred_acceptor = SMILES_functions.get_mordred_descriptors(smiles_acceptor)
df_mordred_acceptor.columns = [f"mordred_acceptor_{c}" for c in df_mordred_acceptor.columns]

df_mordred_donor = SMILES_functions.get_mordred_descriptors(smiles_donor)
df_mordred_donor.columns = [f"mordred_donor_{c}" for c in df_mordred_donor.columns]

# concatenantion Smiles acceptor, donor and input numerical Data
df_mordred = pd.concat([df_mordred_acceptor,df_mordred_donor, X_scaled], axis=1)

# Dropping uncomplete lines
df_mordred = df_mordred.dropna()
df_mordred_output = y_scaled.loc[df_mordred.index]

# Save in csv file
df_mordred.to_csv("Data_Mordred$.csv", index=False, sep=';')
df_mordred_output.to_csv("Output_Mordred.csv", index=False, sep=";")


# Morgan Fingerprints
print("Morgan Fingerprints")
morgan_matrix_acceptor = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_acceptor])
df_morgan_acceptor = pd.DataFrame(morgan_matrix_acceptor, index=df_full.index, columns=[f"morgan_acceptor_{i}" for i in range(morgan_matrix_acceptor.shape[1])])

morgan_matrix_donor = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_donor])
df_morgan_donor = pd.DataFrame(morgan_matrix_donor, index=df_full.index, columns=[f"morgan_donor_{i}" for i in range(morgan_matrix_donor.shape[1])])

# concatenantion Smiles acceptor, donor and input numerical Data
df_morgan = pd.concat([df_morgan_acceptor, df_morgan_donor, X_scaled], axis=1)

# Dropping uncomplete lines
df_morgan = df_morgan.dropna()
df_morgan_output = y_scaled.loc[df_morgan.index]

# Save in csv file
df_morgan.to_csv("Data_Morgan.csv", index=False, sep=';')
df_morgan_output.to_csv("Output_Morgan.csv", index=False, sep=";")

# MACCS Keys fingerprints
print("MACCS Keys Fingerprints")
maccs_matrix_acceptor = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_acceptor])
df_maccs_acceptor = pd.DataFrame(maccs_matrix_acceptor, index=df_full.index, columns=[f"maccs_acceptor_{i}" for i in range(maccs_matrix_acceptor.shape[1])])

maccs_matrix_donor = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_donor])
df_maccs_donor = pd.DataFrame(maccs_matrix_donor, index=df_full.index, columns=[f"maccs_donor_{i}" for i in range(maccs_matrix_donor.shape[1])])

# concatenantion Smiles acceptor, donor and input numerical Data
df_maccs = pd.concat([df_maccs_acceptor, df_maccs_donor, X_scaled], axis=1)

# Dropping uncomplete lines
df_maccs = df_maccs.dropna()
df_maccs_output = y_scaled.loc[df_maccs.index]

# Save in csv file
df_maccs.to_csv("Data_MACCS.csv", index=False, sep=';')
df_maccs_output.to_csv("Output_MACCS.csv", index=False, sep=";")

#PubChem Fingerprints
print("PubChem Fingerprints")
pubchem_matrix_acceptor = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_acceptor])
df_pubchem_acceptor = pd.DataFrame(pubchem_matrix_acceptor, index=df_full.index, columns=[f"pubchem_acceptor_{i}" for i in range(pubchem_matrix_acceptor.shape[1])])

pubchem_matrix_donor = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_donor])
df_pubchem_donor = pd.DataFrame(pubchem_matrix_donor, index=df_full.index, columns=[f"pubchem_donor_{i}" for i in range(pubchem_matrix_donor.shape[1])])

# concatenantion Smiles acceptor, donor and input numerical Data
df_pubchem = pd.concat([df_pubchem_acceptor, df_pubchem_donor, X_scaled], axis=1)

# Dropping uncomplete lines
df_pubchem = df_pubchem.dropna()
df_pubchem_output = y_scaled.loc[df_pubchem.index]

# Save in csv file
df_pubchem.to_csv("Data_PubChem.csv", index=False, sep=';')
df_pubchem_output.to_csv("Output_Pubchem.csv", index=False, sep=";")
"""
print("Data Prepraration Finished")