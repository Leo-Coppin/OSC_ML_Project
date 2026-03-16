import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import SMILES_functions


# Opening CSV File
df = pd.read_csv(
    "Data.csv",
    sep=";",
    dtype=str
)

# Choosen columns  : SMILES code, HOMO, LUMO, EgCV, λ absorption for Donor and Acceptor 
# In this dataset we’ll keep the columns Voc, Jsc, FF and PCE as outputs.
choosen_columns = ['SMILES_acc', 'SMILES_don', 'Voc', 'Jsc', 'FF', 'PCE', 'HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption', 'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption']
numerical_columns =  ['Voc', 'Jsc', 'FF', 'PCE', 'HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption', 'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption']

df_full = df[choosen_columns]

# without SMILES acceptor and SMILES donor -> only numerical columns
df_numerical = df[numerical_columns]


# Decimal point uniformisation : , -> .
df_numerical = df_numerical.replace(",", ".", regex=True)

# Convert type to numeric if not the case
#df_numerical = df_numerical.apply(pd.to_numeric, errors="ignore")

'''

Put preprocessing of numerical Data here 

'''

# Keep only valid rows in numerical data (drop rows with NaN)
df_numerical = df_numerical.dropna()

# To keep fingerprint alignment: select the same valid indices in df_full
df_full = df_full.loc[df_numerical.index]


# spliting inputs and outputs from df_numerical
X_inputs = df_numerical[['HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption',
                         'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption']]

y_outputs = df_numerical[['Voc', 'Jsc', 'FF', 'PCE']]


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


# SMILES code processing
smiles_acceptor = df_full['SMILES_acc']
smiles_donor = df_full['SMILES_don']

lenght_columns = len(smiles_acceptor)

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

df_rdkit = pd.concat([df_rdkit_acceptor, df_rdkit_donor], axis=1)
# concatenantion with input numerical Data

# Dropping uncomplete lines
df_rdkit = df_rdkit.dropna()

# Save in csv file
df_rdkit.to_csv("Data_RDKit.csv", index=False, sep=';')

# Mordred Descriptors
#print("Mordred Descriptors")
#df_mordred_acceptor = SMILES_functions.get_mordred_descriptors(smiles_acceptor)
#df_mordred_acceptor.columns = [f"mordred_acceptor_{c}" for c in df_mordred_acceptor.columns]

#df_mordred_donor = SMILES_functions.get_mordred_descriptors(smiles_donor)
#df_mordred_donor.columns = [f"mordred_donor_{c}" for c in df_mordred_donor.columns]

#df_mordred = pd.concat([df_mordred_acceptor,df_mordred_donor], axis=1)
# concatenantion with input numerical Data

# Dropping uncomplete lines
#df_mordred = df_mordred.dropna()

# Save in csv file
#df_mordred.to_csv("Data_Mordred.csv", index=False, sep=';')

# Morgan Fingerprints
print("Morgan Fingerprints")
morgan_matrix_acceptor = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_acceptor])
df_morgan_acceptor = pd.DataFrame(morgan_matrix_acceptor, index=df_full.index, columns=[f"morgan_acceptor_{i}" for i in range(morgan_matrix_acceptor.shape[1])])

morgan_matrix_donor = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_donor])
df_morgan_donor = pd.DataFrame(morgan_matrix_donor, index=df_full.index, columns=[f"morgan_donor_{i}" for i in range(morgan_matrix_donor.shape[1])])

df_morgan = pd.concat([df_morgan_acceptor, df_morgan_donor], axis=1)
# concatenantion with input numerical Data

# Dropping uncomplete lines
df_morgan = df_morgan.dropna()

# Save in csv file
df_morgan.to_csv("Data_Morgan.csv", index=False, sep=';')


# MACCS Keys fingerprints
print("MACCS Keys Fingerprints")
maccs_matrix_acceptor = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_acceptor])
df_maccs_acceptor = pd.DataFrame(maccs_matrix_acceptor, index=df_full.index, columns=[f"maccs_acceptor_{i}" for i in range(maccs_matrix_acceptor.shape[1])])

maccs_matrix_donor = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_donor])
df_maccs_donor = pd.DataFrame(maccs_matrix_donor, index=df_full.index, columns=[f"maccs_donor_{i}" for i in range(maccs_matrix_donor.shape[1])])


df_maccs = pd.concat([df_maccs_acceptor, df_maccs_donor], axis=1)
# concatenantion with input numerical Data

# Dropping uncomplete lines
df_maccs = df_maccs.dropna()

# Save in csv file
df_maccs.to_csv("Data_MACCS.csv", index=False, sep=';')

#PubChem Fingerprints
print("PubChem Fingerprints")
pubchem_matrix_acceptor = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_acceptor])
df_pubchem_acceptor = pd.DataFrame(pubchem_matrix_acceptor, index=df_full.index, columns=[f"pubchem_acceptor_{i}" for i in range(pubchem_matrix_acceptor.shape[1])])

pubchem_matrix_donor = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_donor])
df_pubchem_donor = pd.DataFrame(pubchem_matrix_donor, index=df_full.index, columns=[f"pubchem_donor_{i}" for i in range(pubchem_matrix_donor.shape[1])])

df_pubchem = pd.concat([df_pubchem_acceptor, df_pubchem_donor], axis=1)
# concatenantion with input numerical Data

# Dropping uncomplete lines
df_pubchem = df_pubchem.dropna()

# Save in csv file
df_pubchem.to_csv("Data_PubChem.csv", index=False, sep=';')


print("Data Prepraration Finished")
