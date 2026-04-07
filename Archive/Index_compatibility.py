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
df = pd.read_csv("Data.csv", sep=";")

# 1. On identifie les lignes techniquement valides pour le GNN
print("🔍 Vérification de la compatibilité des SMILES...")
df['valid'] = df.apply(check_validity, axis=1)
df_clean = df[df['valid'] == True].copy()

# 2. On croise avec tes fichiers de descripteurs existants (Intersection)
# On charge l'index d'un de tes CSV (ex: MACCS) pour être sûr qu'ils ont aussi ces lignes
df_maccs = pd.read_csv("Data_MACCS.csv", sep=";")
common_index = df_clean.index.intersection(df_maccs.index)
df_final = df_clean.loc[common_index]

# 3. Création du Split par Donneur 
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df_final, groups=df_final['SMILES_don']))

split_df = pd.DataFrame(index=df_final.index)
split_df['set'] = 'train'
split_df.loc[df_final.index[test_idx], 'set'] = 'test'

# 4. Sauvegarde du fichier de référence
split_df.to_csv("master_split.csv", sep=";")
print(f"✅ Terminé ! {len(df_final)} molécules synchronisées. Fichier 'master_split.csv' créé.")