import pandas as pd
from rdkit import Chem
from SMILES_to_Graph import smiles_to_graph 
from sklearn.model_selection import GroupShuffleSplit

def check_validity(row):
    try:
        # Test 1: Est-ce que le Graphe est générable ?
        g_don = smiles_to_graph(row['SMILES_don'])
        g_acc = smiles_to_graph(row['SMILES_acc'])
        if g_don is None or g_acc is None: return False
        
        # Test 2: Est-ce que RDKit arrive à lire la molécule ?
        if Chem.MolFromSmiles(row['SMILES_don']) is None: return False
        if Chem.MolFromSmiles(row['SMILES_acc']) is None: return False
        
        return True
    except:
        return False

# Chargement du fichier source
df = pd.read_csv("Data_scaled.csv", sep=";")

# 1. On identifie les lignes techniquement valides pour le GNN
print("🔍 Vérification de la compatibilité des SMILES...")
df['valid'] = df.apply(check_validity, axis=1)
df_clean = df[df['valid'] == True].copy()

# 2. On croise avec tes fichiers de descripteurs existants (Intersection)
# On charge l'index de tous les CSV pour être sûr qu'ils ont aussi ces lignes
df_rdkit = pd.read_csv("Data_RDKit.csv", sep=";")
df_mordred = pd.read_csv("Data_Mordred.csv", sep=";")
df_morgan = pd.read_csv("Data_Morgan.csv", sep=";")
df_maccs = pd.read_csv("Data_MACCS.csv", sep=";")
df_pubchem = pd.read_csv("Data_PubChem.csv", sep=";")

common_index = df_clean.index.intersection(df_rdkit.index)
df_clean = df_clean.loc[common_index]
common_index = df_clean.index.intersection(df_mordred.index)
df_clean = df_clean.loc[common_index]
common_index = df_clean.index.intersection(df_morgan.index)
df_clean = df_clean.loc[common_index]
common_index = df_clean.index.intersection(df_maccs.index)
df_clean = df_clean.loc[common_index]
common_index = df_clean.index.intersection(df_pubchem.index)
df_final = df_clean.loc[common_index]

# 3. Création du Split par Donneur 
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
train_idx, test_idx = next(gss.split(df_final, groups=df_final['SMILES_don']))

split_df = pd.DataFrame(index=df_final.index)
split_df['set'] = 'train'
split_df.loc[df_final.index[test_idx], 'set'] = 'test'

# 4. Sauvegarde du fichier de référence
split_df.to_csv("master_split.csv", sep=";")
print(f"✅ Terminé ! {len(df_final)} molécules synchronisées. Fichier 'master_split.csv' créé.")

data = pd.read_csv("Data_scaled.csv", sep=";")
features = ["SMILES_don", "SMILES_acc", "HOMO_A","LUMO_A","HOMO_D","LUMO_D","Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]

data = data[features]
train = data.loc[split_df[split_df['set'] == 'train'].index]
test  = data.loc[split_df[split_df['set'] == 'test'].index]

train.to_csv("train_dataset.csv", index=False, sep=";")
test.to_csv("test_dataset.csv", index=False, sep=";")