import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

X = pd.read_csv("Data_Compatibility_score.csv", sep=';')
y = pd.read_csv("Output_Compatibility_score.csv", sep=';')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. On entraîne le modèle MultiOutput avec les paramètres qui marchaient pour le PCE
best_base = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=100, subsample=0.8, random_state=42)
multi_model = MultiOutputRegressor(best_base)
multi_model.fit(X_train, y_train)

# 2. On fait la prédiction sur le test
y_pred = multi_model.predict(X_test)

# 3. On calcule le R² pour CHAQUE sortie séparément
# 'raw_values' permet d'avoir un score par colonne au lieu de la moyenne
individual_r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Affichage propre
for name, score in zip(y.columns, individual_r2):
    print(f"Score R² pour {name} : {score:.3f}")
    
# 4. Feature importances par target
feature_names = X.columns.tolist()
target_names = y.columns.tolist()

importances_per_target = pd.DataFrame(
    [est.feature_importances_ for est in multi_model.estimators_],
    index=target_names,
    columns=feature_names
)

print("\nImportance des features par target :")
print(importances_per_target.round(3))

# 5. Importance moyenne sur les 4 targets
mean_importance = importances_per_target.mean(axis=0).sort_values(ascending=False)

print("\nImportance moyenne (triée) :")
print(mean_importance.round(3))