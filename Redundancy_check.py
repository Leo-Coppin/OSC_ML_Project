import pandas as pd
import numpy as np

df = pd.read_csv("fichier.csv")

corr = df.corr(numeric_only=True)

# seuil
threshold = 0.8

# liste des colonnes à supprimer
cols_to_drop = set()

for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            colname = corr.columns[i]
            cols_to_drop.add(colname)

# suppression
df_reduced = df.drop(columns=list(cols_to_drop))

print("Variables supprimées :", len(cols_to_drop))
print("Liste :", cols_to_drop)
