import pandas as pd

# Opening CSV File
df = pd.read_csv(
    "Data_Final_Article1_Bandgap_engineering 1_Original.csv",
    sep=";",
    dtype=str
)

# Choosen columns  : SMILES code, HOMO, LUMO, EgCV, λ absorption for Donor and Acceptor 
# In this dataset we’ll keep the columns Voc, Jsc, FF and PCE as outputs.
choosen_columns = ['SMILES_acc', 'SMILES_don', 'Voc', 'Jsc', 'FF', 'PCE', 'HOMO_A', 'LUMO_A', 'EgCV_A', 'λ_A_absorption', 'HOMO_D', 'LUMO_D', 'EgCV_D', 'λ_D_absorption']


df_full = df[choosen_columns]

# without SMILES acceptor and SMILES donor -> only numerical columns
df_numerical = df_full.drop(['SMILES_acc', 'SMILES_don'])


# Decimal point uniformisation : , -> .
df_numerical = df_numerical.replace(",", ".", regex=True)

# Convert type to numeric if not the case
df_numerical = df_numerical.apply(pd.to_numeric, errors="ignore")

# SMILES code processing





Data = df.dropna()