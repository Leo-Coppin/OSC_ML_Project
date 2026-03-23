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