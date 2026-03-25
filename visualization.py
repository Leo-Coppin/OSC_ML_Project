import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X = pd.read_csv("Data_RDKit.csv", sep=";")
y = pd.read_csv("Output_RDKit.csv", sep=";")

df = pd.concat([X, y], axis=1)

sns.histplot(df["scaled_PCE"], bins=30, kde=True)
plt.title("Distribution of PCE")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

sns.scatterplot(x=df["HOMO_D"], y=df["scaled_PCE"])
plt.title("HOMO vs PCE")
plt.show()

sns.scatterplot(x=df["LUMO_D"], y=df["scaled_PCE"])
plt.title("LUMO vs PCE")
plt.show()