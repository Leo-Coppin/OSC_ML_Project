"""
Conversion SMILES → Graphes moléculaires
Utilise : RDKit + PyTorch Geometric
Dataset : CSV avec colonnes SMILES_don, SMILES_acc, PCE, Voc, Jsc, FF, ΔHOMO, ΔLUMO
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np


# =============================================================================
# 1. FONCTIONS D'EXTRACTION DES FEATURES
# =============================================================================

def get_atom_features(atom):
    """
    Extrait les features d'un atome (nœud du graphe).
    Retourne un vecteur numérique pour chaque atome.
    """

    # Type d'hybridation possible
    hybridization_types = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]

    features = [
        atom.GetAtomicNum(),                        # Numéro atomique (ex: 6 pour C, 7 pour N)
        atom.GetDegree(),                           # Nombre de liaisons avec d'autres atomes
        atom.GetNumImplicitHs(),                    # Nombre d'hydrogènes implicites
        int(atom.GetIsAromatic()),                  # Aromatique ? (1 ou 0)
        int(atom.IsInRing()),                       # Dans un cycle ? (1 ou 0)
        atom.GetFormalCharge(),                   # Charge formelle (ex: -1, 0, +1)
        # Encodage one-hot de l'hybridation
        *[int(atom.GetHybridization() == h) for h in hybridization_types],
    ]

    return features


def get_bond_features(bond):
    """
    Extrait les features d'une liaison (arête du graphe).
    Retourne un vecteur numérique pour chaque liaison.
    """

    bond_types = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]

    features = [
        # Encodage one-hot du type de liaison
        *[int(bond.GetBondType() == bt) for bt in bond_types],
        int(bond.GetIsConjugated()),   # Liaison conjuguée ? (1 ou 0)
        int(bond.IsInRing()),          # Dans un cycle ? (1 ou 0)
    ]

    return features


# =============================================================================
# 2. CONVERSION D'UN SMILES EN GRAPHE PyTorch Geometric
# =============================================================================

def smiles_to_graph(smiles):
    """
    Convertit un SMILES en objet torch_geometric.data.Data.

    Retourne :
        Data(
            x          = features des atomes  [num_atoms, num_atom_features]
            edge_index = indices des liaisons  [2, num_edges * 2]  (bidirectionnel)
            edge_attr  = features des liaisons [num_edges * 2, num_bond_features]
        )
        Retourne None si le SMILES est invalide.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"⚠️  SMILES invalide : {smiles}")
        return None

    # --- Features des atomes (nœuds) ---
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # --- Features des liaisons (arêtes) ---
    # PyTorch Geometric représente chaque liaison dans les 2 sens (A→B et B→A)
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Liaison dans les deux sens
        edge_indices += [[i, j], [j, i]]
        edge_features += [bond_feat, bond_feat]

    if len(edge_indices) == 0:
        # Molécule avec un seul atome, pas de liaisons
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# 3. CHARGEMENT DU DATASET ET CONSTRUCTION DES GRAPHES
# =============================================================================

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='warn')

    # Conversion forcée des colonnes numériques (gère les virgules ET les points)
    cols_to_convert = ['HOMO_A', 'LUMO_A', 'HOMO_D', 'LUMO_D', 'PCE', 'Voc', 'Jsc', 'FF']
    for col in cols_to_convert:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calcul de ΔHOMO et ΔLUMO
    df['ΔHOMO'] = df['HOMO_A'] - df['HOMO_D']
    df['ΔLUMO'] = df['LUMO_A'] - df['LUMO_D']

    target_columns = ['PCE', 'Voc', 'Jsc', 'FF', 'ΔHOMO', 'ΔLUMO']

    # Supprime les lignes avec des valeurs manquantes
    df = df.dropna(subset=['SMILES_don', 'SMILES_acc'] + target_columns)

    dataset = []
    skipped = 0

    for idx, row in df.iterrows():
        graph_don = smiles_to_graph(row['SMILES_don'])
        graph_acc = smiles_to_graph(row['SMILES_acc'])

        if graph_don is None or graph_acc is None:
            skipped += 1
            continue

        targets = torch.tensor(
            [float(row[col]) for col in target_columns],
            dtype=torch.float
        )

        dataset.append({
            'graph_donor':    graph_don,
            'graph_acceptor': graph_acc,
            'y':              targets,
            'smiles_don':     row['SMILES_don'],
            'smiles_acc':     row['SMILES_acc'],
        })

    print(f"✅ Dataset chargé : {len(dataset)} paires valides ({skipped} ignorées)")
    return dataset

# =============================================================================
# 4. VÉRIFICATION RAPIDE
# =============================================================================

def verify_sample(dataset, index=0):
    """Affiche les détails d'un exemple pour vérifier que tout est correct."""

    sample = dataset[index]
    don = sample['graph_donor']
    acc = sample['graph_acceptor']

    print("\n--- Exemple n°", index, "---")
    print(f"SMILES donneur  : {sample['smiles_don']}")
    print(f"SMILES accepteur: {sample['smiles_acc']}")
    print(f"\nGraphe DONNEUR")
    print(f"  Nœuds (atomes)   : {don.x.shape}  → {don.x.shape[0]} atomes, {don.x.shape[1]} features")
    print(f"  Arêtes (liaisons): {don.edge_index.shape[1] // 2} liaisons ({don.edge_index.shape[1]} avec les 2 sens)")
    print(f"  Features arêtes  : {don.edge_attr.shape}")
    print(f"\nGraphe ACCEPTEUR")
    print(f"  Nœuds (atomes)   : {acc.x.shape}  → {acc.x.shape[0]} atomes, {acc.x.shape[1]} features")
    print(f"  Arêtes (liaisons): {acc.edge_index.shape[1] // 2} liaisons ({acc.edge_index.shape[1]} avec les 2 sens)")
    print(f"  Features arêtes  : {acc.edge_attr.shape}")
    print(f"\nCibles (y) : {sample['y']}")
    print(f"  → PCE={sample['y'][0]:.3f}, Voc={sample['y'][1]:.3f}, Jsc={sample['y'][2]:.3f}")
    print(f"  → FF={sample['y'][3]:.3f}, ΔHOMO={sample['y'][4]:.3f}, ΔLUMO={sample['y'][5]:.3f}")
    print(f"\nFeatures du 1er atome du donneur : {don.x[0].tolist()}")
    print(f"  → Feature de la première liaison : {don.edge_attr[0].tolist()}")


   


# =============================================================================
# 5. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":

    CSV_PATH  = "CSV_PATH" #raw csv datafile 

    # Chargement et conversion
    dataset = load_dataset(CSV_PATH)

    # Vérification sur le premier exemple
    verify_sample(dataset, index=0)
   

    
  
