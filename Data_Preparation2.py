import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import SMILES_functions
from sklearn.model_selection import GroupShuffleSplit

import pandas as pd
import numpy as np
from rdkit import Chem
# from rdkit.Chem import AllChem, MACCSkeys, Descriptors
# from mordred import Calculator, descriptors

from SMILES_to_Graph import load_dataset, smiles_to_graph
import SMILES_functions

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

df = df[choosen_columns]

# without SMILES acceptor and SMILES donor -> only numerical columns
df_numerical = df[numerical_columns]

# Decimal point uniformisation : , -> .
df_numerical = df_numerical.replace(",", ".", regex=True)

#extract numbers 
for col in df_numerical.columns:
    df_numerical[col] = df_numerical[col].str.extract(r'([0-9.]+)')

df_numerical = df_numerical.apply(pd.to_numeric, errors="coerce")

# Keep only valid rows in numerical data (drop rows with NaN)
df_numerical = df_numerical

# To keep fingerprint alignment: select the same valid indices in df
df = df.loc[df_numerical.index]


# spliting inputs and outputs from df_numerical
y = df_numerical[['Voc', 'Jsc', 'FF', 'PCE']]

y['delta_HOMO'] = df_numerical['HOMO_A'] - df_numerical['HOMO_D']
y['delta_LUMO']  = df_numerical['LUMO_A'] - df_numerical['LUMO_D']


# Scaling outputs
scaler_outputs = StandardScaler()
y_scaled = pd.DataFrame(
    scaler_outputs.fit_transform(y),
    columns=[f"{col}" for col in y.columns],
    index=df_numerical.index
)


smiles_codes = df[['SMILES_acc', 'SMILES_don']]
Data_scaled = pd.concat([smiles_codes, y_scaled], axis =1)

Data_scaled.to_csv("Data_scaled.csv", index=False, sep=";")

Data_Models = pd.concat([smiles_codes,df_numerical['HOMO_A'], df_numerical['HOMO_D'], df_numerical['LUMO_A'], df_numerical['LUMO_D'] ,y_scaled], axis=1)
"""
good_samples = []
for idx, sample in Data_Models.iterrows() : 
    sample_transform = True
    smile_acceptor = sample['SMILES_acc']
    smile_donor = sample['SMILES_don']
    
    #check graph
    g_don = smiles_to_graph(smile_acceptor)
    g_acc = smiles_to_graph(smile_donor)
    if g_don is None or g_acc is None: 
        sample_transform = False
    else : 
        #check mordred, morgan and MACCS Keys
        acceptor = SMILES_functions.smiles_to_mol(smile_acceptor)
        donor = SMILES_functions.smiles_to_mol(smile_donor)
        
        if acceptor is None or donor is None: 
            sample_transform = False
        else : 
            #check rdkit   
            rd_acc = SMILES_functions.get_rdkit_descriptors(smile_acceptor)
            rd_don = SMILES_functions.get_rdkit_descriptors(smile_donor)
            
            if rd_acc is None or rd_don is None:
                continue 
            
            has_nan_acc = any(pd.isna(v) for v in rd_acc.values())
            has_nan_don = any(pd.isna(v) for v in rd_don.values())
            
            if has_nan_acc or has_nan_don:
                sample_transform = False
            
    if sample_transform :
        good_samples.append(sample)
    
    
good_samples = pd.DataFrame(good_samples, columns=Data_Models.columns)
print(good_samples)

# 3. Création du Split par Donneur 
gss = GroupShuffleSplit(n_splits=1, test_size=0.29, random_state=42)
train_idx, test_idx = next(gss.split(good_samples, groups=good_samples['SMILES_don']))

split_df = pd.DataFrame(index=good_samples.index)
split_df['set'] = 'train'
split_df.loc[good_samples.index[test_idx], 'set'] = 'test'

#make into the different csv files

train = good_samples.loc[split_df[split_df['set'] == 'train'].index]
test  = good_samples.loc[split_df[split_df['set'] == 'test'].index]

train.to_csv("train_dataset.csv", index=False, sep=";")
test.to_csv("test_dataset.csv", index=False, sep=";")
"""
train = pd.read_csv("train_dataset.csv", sep=";")
test = pd.read_csv("test_dataset.csv", sep=";")

smiles_acceptor_train = train['SMILES_acc']
smiles_acceptor_test = test['SMILES_acc']
smiles_donor_train = train['SMILES_don']
smiles_donor_test = test['SMILES_don']  

# # RDKit 
# rdkit_list_acceptor_train = []
# rdkit_list_acceptor_test = []
# rdkit_list_donor_train = []
# rdkit_list_donor_test = []

# for smile in smiles_acceptor_train :
#     rdkit_list_acceptor_train.append(SMILES_functions.get_rdkit_descriptors(smile))
# for smile in smiles_acceptor_test :
#     rdkit_list_acceptor_test.append(SMILES_functions.get_rdkit_descriptors(smile))
# for smile in smiles_donor_train : 
#     rdkit_list_donor_train.append(SMILES_functions.get_rdkit_descriptors(smile))
# for smile in smiles_donor_test : 
#     rdkit_list_donor_test.append(SMILES_functions.get_rdkit_descriptors(smile))
    
# df_rdkit_acceptor_train = pd.DataFrame(rdkit_list_acceptor_train, index=train.index)
# df_rdkit_acceptor_train.columns=[f"rdkit_acceptor_{c}" for c in df_rdkit_acceptor_train.columns]
# df_rdkit_donor_train = pd.DataFrame(rdkit_list_donor_train, index=train.index)
# df_rdkit_donor_train.columns=[f"rdkit_donor_{c}" for c in df_rdkit_donor_train.columns]

# df_rdkit_acceptor_test = pd.DataFrame(rdkit_list_acceptor_test, index=test.index)
# df_rdkit_acceptor_test.columns=[f"rdkit_acceptor_{c}" for c in df_rdkit_acceptor_test.columns]
# df_rdkit_donor_test = pd.DataFrame(rdkit_list_donor_test, index=test.index)
# df_rdkit_donor_test.columns=[f"rdkit_donor_{c}" for c in df_rdkit_donor_test.columns]

# df_rdkit_train = pd.concat([df_rdkit_acceptor_train, df_rdkit_donor_train], axis=1)#.dropna()
# df_rdkit_test = pd.concat([df_rdkit_acceptor_test, df_rdkit_donor_test], axis=1)#.dropna()

# df_rdkit_train.to_csv("Data_RDKit_train.csv", index=False, sep=';')
# df_rdkit_test.to_csv("Data_RDKit_test.csv", index=False, sep=';')

# mordred
print("Mordred Descriptors")
df_mordred_acc_train = SMILES_functions.get_mordred_descriptors(smiles_acceptor_train)
df_mordred_don_train = SMILES_functions.get_mordred_descriptors(smiles_donor_train)
df_mordred_acc_train.columns = [f"mordred_acceptor_{c}" for c in df_mordred_acc_train.columns]
df_mordred_don_train.columns = [f"mordred_donor_{c}" for c in df_mordred_don_train.columns]

df_mordred_acc_test = SMILES_functions.get_mordred_descriptors(smiles_acceptor_test)
df_mordred_don_test = SMILES_functions.get_mordred_descriptors(smiles_donor_test)
df_mordred_acc_test.columns = [f"mordred_acceptor_{c}" for c in df_mordred_acc_test.columns]
df_mordred_don_test.columns = [f"mordred_donor_{c}" for c in df_mordred_don_test.columns]

df_mordred_train = pd.concat([df_mordred_acc_train, df_mordred_don_train], axis=1).dropna()
df_mordred_test = pd.concat([df_mordred_acc_test, df_mordred_don_test], axis=1).dropna()

df_mordred_train.to_csv("Data_Mordred_train.csv", index=False, sep=';')
df_mordred_test.to_csv("Data_Mordred_test.csv", index=False, sep=';')

# Morgan
# print("Morgan Fingerprints")
# morgan_matrix_acc_train = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_acceptor_train])
# df_morgan_acc_train = pd.DataFrame(morgan_matrix_acc_train, index=train.index, columns=[f"morgan_acc_{i}" for i in range(morgan_matrix_acc_train.shape[1])])
# morgan_matrix_don_train = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_donor_train])
# df_morgan_don_train = pd.DataFrame(morgan_matrix_don_train, index=train.index, columns=[f"morgan_don_{i}" for i in range(morgan_matrix_don_train.shape[1])])

# morgan_matrix_acc_test = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_acceptor_test])
# df_morgan_acc_test = pd.DataFrame(morgan_matrix_acc_test, index=test.index, columns=[f"morgan_acc_{i}" for i in range(morgan_matrix_acc_test.shape[1])])
# morgan_matrix_don_test = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in smiles_donor_test])
# df_morgan_don_test = pd.DataFrame(morgan_matrix_don_test, index=test.index, columns=[f"morgan_don_{i}" for i in range(morgan_matrix_don_test.shape[1])])

# df_morgan_train = pd.concat([df_morgan_acc_train, df_morgan_don_train], axis=1).dropna()
# df_morgan_test = pd.concat([df_morgan_acc_test, df_morgan_don_test], axis=1).dropna()

# df_morgan_train.to_csv("Data_Morgan_train.csv", index=False, sep=';')
# df_morgan_test.to_csv("Data_Morgan_test.csv", index=False, sep=';')

# MACCS
# print("MACCS Keys Fingerprints")
# maccs_matrix_acc_train = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_acceptor_train])
# df_maccs_acc_train = pd.DataFrame(maccs_matrix_acc_train, index=train.index, columns=[f"maccs_acc_{i}" for i in range(maccs_matrix_acc_train.shape[1])])
# maccs_matrix_don_train = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_donor_train])
# df_maccs_don_train = pd.DataFrame(maccs_matrix_don_train, index=train.index, columns=[f"maccs_don_{i}" for i in range(maccs_matrix_don_train.shape[1])])

# maccs_matrix_acc_test = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_acceptor_test])
# df_maccs_acc_test = pd.DataFrame(maccs_matrix_acc_test, index=test.index, columns=[f"maccs_acc_{i}" for i in range(maccs_matrix_acc_test.shape[1])])
# maccs_matrix_don_test = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in smiles_donor_test])
# df_maccs_don_test = pd.DataFrame(maccs_matrix_don_test, index=test.index, columns=[f"maccs_don_{i}" for i in range(maccs_matrix_don_test.shape[1])])

# df_maccs_train = pd.concat([df_maccs_acc_train, df_maccs_don_train], axis=1).dropna()
# df_maccs_test = pd.concat([df_maccs_acc_test, df_maccs_don_test], axis=1).dropna()

# df_maccs_train.to_csv("Data_MACCS_train.csv", index=False, sep=';')
# df_maccs_test.to_csv("Data_MACCS_test.csv", index=False, sep=";")

# # PubChem 
# print("PubChem Fingerprints")
# pc_matrix_acc_train = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_acceptor_train])
# df_pc_acc_train = pd.DataFrame(pc_matrix_acc_train, index=train.index, columns=[f"pubchem_acc_{i}" for i in range(pc_matrix_acc_train.shape[1])])
# pc_matrix_don_train = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_donor_train])
# df_pc_don_train = pd.DataFrame(pc_matrix_don_train, index=train.index, columns=[f"pubchem_don_{i}" for i in range(pc_matrix_don_train.shape[1])])

# pc_matrix_acc_test = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_acceptor_test])
# df_pc_acc_test = pd.DataFrame(pc_matrix_acc_test, index=test.index, columns=[f"pubchem_acc_{i}" for i in range(pc_matrix_acc_test.shape[1])])
# pc_matrix_don_test = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in smiles_donor_test])
# df_pc_don_test = pd.DataFrame(pc_matrix_don_test, index=test.index, columns=[f"pubchem_don_{i}" for i in range(pc_matrix_don_test.shape[1])])

# df_pubchem_train = pd.concat([df_pc_acc_train, df_pc_don_train], axis=1).dropna()
# df_pubchem_test = pd.concat([df_pc_acc_test, df_pc_don_test], axis=1).dropna()

# df_pubchem_train.to_csv("Data_PubChem_train.csv", index=False, sep=';')
# df_pubchem_test.to_csv("Data_PubChem_test.csv", index=False, sep=';')




# #test graphs -> validated
# good_samples.to_csv("csv_test.csv", index=False, sep=";")
# test = load_dataset("csv_test.csv")

# test RDKit -> problems 
# rdkit_list_acceptor = []
# rdkit_list_donor = []
# for smile in good_samples['SMILES_acc'] :
#     rdkit_list_acceptor.append(SMILES_functions.get_rdkit_descriptors(smile))
# for smile in good_samples['SMILES_don'] : 
#     rdkit_list_donor.append(SMILES_functions.get_rdkit_descriptors(smile))
    
# df_rdkit_acceptor = pd.DataFrame(rdkit_list_acceptor, index=good_samples.index)
# df_rdkit_acceptor.columns=[f"rdkit_acceptor_{c}" for c in df_rdkit_acceptor.columns]
# df_rdkit_donor = pd.DataFrame(rdkit_list_donor, index=good_samples.index)
# df_rdkit_donor.columns=[f"rdkit_donor_{c}" for c in df_rdkit_donor.columns]
# df_rdkit = pd.concat([df_rdkit_acceptor, df_rdkit_donor], axis=1).dropna()
# print(f"RDKit descriptors calculés pour {len(df_rdkit)} échantillons valides.") 

# test mordred
# df_mordred_acc = SMILES_functions.get_mordred_descriptors(good_samples['SMILES_acc'])
# df_mordred_don = SMILES_functions.get_mordred_descriptors(good_samples['SMILES_don'])
# df_mordred_acc.columns = [f"mordred_acceptor_{c}" for c in df_mordred_acc.columns]
# df_mordred_don.columns = [f"mordred_donor_{c}" for c in df_mordred_don.columns]

# df_mordred = pd.concat([df_mordred_acc, df_mordred_don], axis=1).dropna()
# print(f"Mordred descriptors calculés pour {len(df_mordred)} échantillons valides.")

# # test morgan -> validated 
# morgan_matrix_acc = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in good_samples['SMILES_acc']])
# df_morgan_acc = pd.DataFrame(morgan_matrix_acc, index=good_samples.index, columns=[f"morgan_acc_{i}" for i in range(morgan_matrix_acc.shape[1])])
# morgan_matrix_don = np.vstack([SMILES_functions.get_morgan_fingerprint(s) for s in good_samples['SMILES_don']])
# df_morgan_don = pd.DataFrame(morgan_matrix_don, index=good_samples.index, columns=[f"morgan_don_{i}" for i in range(morgan_matrix_don.shape[1])])

# df_morgan = pd.concat([df_morgan_acc, df_morgan_don], axis=1).dropna()
# print(f"Morgan fingerprints calculés pour {len(df_morgan)} échantillons valides.")

# # test MACCS Keys -> validated 
# maccs_matrix_acc = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in good_samples['SMILES_acc']])
# df_maccs_acc = pd.DataFrame(maccs_matrix_acc, index=good_samples.index, columns=[f"maccs_acc_{i}" for i in range(maccs_matrix_acc.shape[1])])
# maccs_matrix_don = np.vstack([SMILES_functions.get_maccs_fingerprint(s) for s in good_samples['SMILES_don']])
# df_maccs_don = pd.DataFrame(maccs_matrix_don, index=good_samples.index, columns=[f"maccs_don_{i}" for i in range(maccs_matrix_don.shape[1])])

# df_maccs = pd.concat([df_maccs_acc, df_maccs_don], axis=1).dropna()
# print(f"MACCS Keys calculés pour {len(df_maccs)} échantillons valides.")

# test PubChem
# pc_matrix_acc = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in good_samples['SMILES_acc']])
# df_pc_acc = pd.DataFrame(pc_matrix_acc, index=good_samples.index, columns=[f"pubchem_acc_{i}" for i in range(pc_matrix_acc.shape[1])])
# pc_matrix_don = np.vstack([SMILES_functions.get_pubchem_fingerprint(s) for s in good_samples['SMILES_don']])
# df_pc_don = pd.DataFrame(pc_matrix_don, index=good_samples.index, columns=[f"pubchem_don_{i}" for i in range(pc_matrix_don.shape[1])])

# df_pubchem = pd.concat([df_pc_acc, df_pc_don], axis=1).dropna()
# print(f"PubChem fingerprints calculés pour {len(df_pubchem)} échantillons valides.")