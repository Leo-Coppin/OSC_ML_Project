import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import SMILES_functions
from sklearn.model_selection import GroupShuffleSplit

import pandas as pd
import numpy as np
from rdkit import Chem

y_train = pd.read_csv("train_dataset.csv", sep=';')
y_test = pd.read_csv("test_dataset.csv", sep=';')



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

X = pd.concat([y_train, y_test], ignore_index=True)
X_RDKit = pd.concat([X_train_RDKit, X_test_RDKit], ignore_index=True)
X_Mordred = pd.concat([X_train_Mordred, X_test_Mordred], ignore_index=True)
X_Morgan = pd.concat([X_train_Morgan, X_test_Morgan], ignore_index=True)
X_MACCS = pd.concat([X_train_MACCS, X_test_MACCS], ignore_index =True)
X_Pubchem = pd.concat([X_train_Pubchem, X_test_Pubchem], ignore_index=True)

seeds = [43, 44, 45, 46, 47, 48, 49]
i = 0
for seed in seeds:
    i +=1 
    gss = GroupShuffleSplit(n_splits=1, test_size=0.29, random_state=seed)
    train_idx, test_idx = next(gss.split(X, groups=X['SMILES_don']))

    split_df = pd.DataFrame(index=X.index)
    split_df['set'] = 'train'
    split_df.loc[X.index[test_idx], 'set'] = 'test'

    #make into the different csv file
    train = X.loc[split_df[split_df['set'] == 'train'].index]
    test  = X.loc[split_df[split_df['set'] == 'test'].index]

    train.to_csv(f"DataShuffle/train_dataset_{i}.csv", index=False, sep=";")
    test.to_csv(f"DataShuffle/test_dataset_{i}.csv", index=False, sep=";")
    
    X_train_RDKit_split = X_RDKit.loc[split_df[split_df['set'] == 'train'].index]
    X_test_RDKit_split = X_RDKit.loc[split_df[split_df['set'] == 'test'].index]
    X_train_RDKit_split.to_csv(f"DataShuffle/Data_RDKit_train_{i}.csv", index=False, sep=";")
    X_test_RDKit_split.to_csv(f"DataShuffle/Data_RDKit_test_{i}.csv", index=False, sep=";")
    
    X_train_Mordred_split = X_Mordred.loc[split_df[split_df['set'] == 'train'].index]
    X_test_Mordred_split = X_Mordred.loc[split_df[split_df['set'] == 'test'].index]
    X_train_Mordred_split.to_csv(f"DataShuffle/Data_Mordred_train_{i}.csv", index=False, sep=";")
    X_test_Mordred_split.to_csv(f"DataShuffle/Data_Mordred_test_{i}.csv", index=False, sep=";") 
    
    X_train_Morgan_split = X_Morgan.loc[split_df[split_df['set'] == 'train'].index]
    X_test_Morgan_split = X_Morgan.loc[split_df[split_df['set'] == 'test'].index]
    X_train_Morgan_split.to_csv(f"DataShuffle/Data_Morgan_train_{i}.csv", index=False, sep=";")
    X_test_Morgan_split.to_csv(f"DataShuffle/Data_Morgan_test_{i}.csv", index=False, sep=";")
    
    X_train_MACCS_split = X_MACCS.loc[split_df[split_df['set'] == 'train'].index]
    X_test_MACCS_split = X_MACCS.loc[split_df[split_df['set'] == 'test'].index]
    X_train_MACCS_split.to_csv(f"DataShuffle/Data_MACCS_train_{i}.csv", index=False, sep=";")
    X_test_MACCS_split.to_csv(f"DataShuffle/Data_MACCS_test_{i}.csv", index=False, sep=";")
    
    X_train_Pubchem_split = X_Pubchem.loc[split_df[split_df['set'] == 'train'].index]
    X_test_Pubchem_split = X_Pubchem.loc[split_df[split_df['set'] == 'test'].index]
    X_train_Pubchem_split.to_csv(f"DataShuffle/Data_PubChem_train_{i}.csv", index=False, sep=";")
    X_test_Pubchem_split.to_csv(f"DataShuffle/Data_PubChem_test_{i}.csv", index=False, sep=";")
