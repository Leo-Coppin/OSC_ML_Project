import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
# On charge ton modèle et tes données
model = joblib.load("best_rf_mordred_optimized.pkl")
X_test = pd.read_csv("Data_Mordred_test.csv", sep=';').apply(pd.to_numeric, errors='coerce').fillna(0)

# Alignement rigoureux des colonnes
cols_at_fit = model.estimators_[0].feature_names_in_
X_test = X_test.reindex(columns=cols_at_fit, fill_value=0)

# Renommage (M_A_ et M_D_) pour que ce soit lisible
new_cols = {col: col.replace('mordred_acceptor_', 'M_A_').replace('mordred_donor_', 'M_D_') for col in X_test.columns}
X_test_renamed = X_test.rename(columns=new_cols)

# =============================================================================
# 2. GÉNÉRATION DU GRAPHIQUE OPTIMISÉ (1 Colorbar + Espace)
# =============================================================================
targets = {"PCE": 3, "delta_HOMO": 4, "delta_LUMO": 5}

# Figure très large (26) pour laisser respirer les 3 graphiques
fig = plt.figure(figsize=(26, 9))

for i, (name, idx) in enumerate(targets.items()):
    print(f"⏳ Calcul SHAP pour : {name}...")
    
    sub_model = model.estimators_[idx]
    explainer = shap.TreeExplainer(sub_model)
    shap_values = explainer.shap_values(X_test_renamed)
    
    # Création du sous-graphique (1 ligne, 3 colonnes)
    ax = fig.add_subplot(1, 3, i+1)
    
    # ASTUCE 1 : colorbar=False supprime la barre répétée
    shap.summary_plot(
        shap_values, 
        X_test_renamed, 
        max_display=12, 
        plot_type="dot", 
        show=False,
        color_bar=False 
    )
    
    # ASTUCE 2 : On force l'axe X pour bien voir en dessous de zéro
    # Ajuste ces valeurs (-0.03 et 0.03) si tes points dépassent encore
    plt.xlim(-0.035, 0.035)
    
    plt.title(f"Influence sur {name}", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel("Impact SHAP", fontsize=14)
    plt.subplots_adjust(left=0.1)

# ASTUCE 3 : Ajout d'une SEULE barre de couleur tout à droite
# [position_gauche, position_bas, largeur, hauteur]
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cmap = shap.plots.colors.red_blue
m = cm.ScalarMappable(cmap=cmap)
m.set_array([0, 1]) # Les valeurs SHAP sont normalisées pour la couleur (0=Low, 1=High)
cb = fig.colorbar(m, cax=cbar_ax, ticks=[0, 1])
cb.set_ticklabels(['Basse', 'Élevée'])
cb.set_label('Valeur du Descripteur', size=14, labelpad=10)

# Ajustement final pour ne pas que les graphiques se marchent dessus
plt.subplots_adjust(wspace=0.4, right=0.90)

plt.savefig("SHAP_Triple_Final_Lisible.png", dpi=300, bbox_inches='tight')
print("✅ Image sauvegardée : SHAP_Triple_Final_Lisible.png")
plt.show()