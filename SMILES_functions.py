import pandas as pd 
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

import warnings
warnings.filterwarnings("ignore")

def smiles_to_mol(smiles):
    if pd.isna(smiles) or str(smiles).strip()=="":
        return None
    return Chem.MolFromSmiles(str(smiles).strip())


# rdkit descriptors -> return dictionnary
def get_rdkit_descriptors(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None : 
        return {name: np.nan for name, _ in Descriptors.descList}
    result = {}
    for name, func in Descriptors.descList:
        try:
            result[name] = func(mol)
        except Exception:
            result[name] = np.nan
    return result 

# Mordred descriptors -> return dataframe 
def get_mordred_descriptors(smiles_column):
    try:
        from mordred import Calculator, descriptors as mordred_desc
    except ImportError:
        print("⚠️  Mordred non installé.")
        return pd.DataFrame(index=smiles_column.index)

    calc = Calculator(mordred_desc, ignore_3D=True)
    
    mols = []
    valid_indices = []
    
    for idx, smiles in smiles_column.items():
        mol = smiles_to_mol(smiles)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(idx)
        else:
            print(f"⚠️  SMILES invalide ignoré à l'index {idx} : {smiles}")
    
    if not mols:
        print("❌ Aucune molécule valide trouvée.")
        return pd.DataFrame(index=smiles_column.index)

    df_mordred = calc.pandas(mols, nproc=1)
    df_mordred.index = valid_indices
    df_mordred = df_mordred.reindex(smiles_column.index)
    df_mordred = df_mordred.apply(pd.to_numeric, errors="coerce")

    # Remplacement des NaN par la médiane
    df_mordred = df_mordred.fillna(df_mordred.median(numeric_only=True))

    return df_mordred
 

# Morgan Fingerprints -> return np_array
def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)

# Maccs keys fingerprints -> return np_array 
def get_maccs_fingerprint(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.uint8)
 

# pubchem fingerprints -> return array
def get_pubchem_fingerprint(smiles):
    #need internet connexion for calls to API pubchem
    try:
        import pubchempy as pcp
    except ImportError:
        print("⚠️  pubchempy non installé. Lancez : pip install pubchempy")
        return np.zeros(881, dtype=np.uint8)
 
    if pd.isna(smiles) or str(smiles).strip() == "":
        return np.zeros(881, dtype=np.uint8)
    try:
        compounds = pcp.get_compounds(str(smiles).strip(), namespace="smiles")
        if not compounds:
            return np.zeros(881, dtype=np.uint8)
        
        fp_hex = compounds[0].cactvs_fingerprint   # chaîne binaire '0110...'
        if fp_hex is None:
            return np.zeros(881, dtype=np.uint8)
        
        fp_array = np.array([int(b) for b in fp_hex], dtype=np.uint8)
        # Assurer longueur 881
        if len(fp_array) < 881:
            fp_array = np.pad(fp_array, (0, 881 - len(fp_array)))
        return fp_array[:881]
    except Exception:
        return np.zeros(881, dtype=np.uint8)
