import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

X = pd.read_csv("Data_Compatibility_score15.csv", sep=';')
y = pd.read_csv("Output_Compatibility_score15.csv", sep=';')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. On entraîne le modèle MultiOutput avec les paramètres qui marchaient pour le PCE
best_base = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=100, subsample=0.8, random_state=42)
multi_model = MultiOutputRegressor(best_base)
multi_model.fit(X_train, y_train)

# 2. On fait la prédiction sur le test
y_pred = multi_model.predict(X_test)

<<<<<<< Updated upstream
# 3. On calcule le R² pour CHAQUE sortie séparément
# 'raw_values' permet d'avoir un score par colonne au lieu de la moyenne
individual_r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Affichage propre
for name, score in zip(y.columns, individual_r2):
    print(f"Score R² pour {name} : {score:.3f}")
=======
# I used grid search to find the best hyperparameters for the Gradient Boosting Regressor, but it is commented out to save time during execution. The best parameters found are used to initialize the base_gbm model.
param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__learning_rate': [0.01, 0.05, 0.1],
    'estimator__max_depth': [3, 4, 5],
    'estimator__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    final_model, 
    param_grid, 
    cv=5, # 5-fold cross-validation
    scoring='r2', # R² score as the evaluation metric, R² for FF is only around 0.29 wereas for the other three it is around 0.5, so it is harder to determine what impact the most the FF factor.
    n_jobs=-1 # Use all available CPU cores for parallel processing
)

grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_

# print(f"Meilleurs paramètres : {grid_search.best_params_}")
# print(f"Meilleur R² moyen : {grid_search.best_score_:.3f}")

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

scaler_outputs = joblib.load("scaler_outputs.pkl")
y_pred_real = scaler_outputs.inverse_transform(y_pred)
y_test_real = scaler_outputs.inverse_transform(y_test)

target_names = ['Voc (V)', 'Jsc (mA/cm²)', 'FF (%)', 'PCE (%)']
df_test_real = pd.DataFrame(y_test_real, columns=target_names)
df_pred_real = pd.DataFrame(y_pred_real, columns=target_names)

print("\n--- MODEL PERFORMANCE (Real Units) ---")
for i, col in enumerate(target_names):
    r2 = r2_score(df_test_real[col], df_pred_real[col])
    mae = mean_absolute_error(df_test_real[col], df_pred_real[col])
    print(f"{col:15} | R² Score: {r2:6.3f} | MAE: {mae:6.3f}")
"""
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

plt.close('all') 

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

fig.suptitle("Experimental vs. Predicted Values (Organic Solar Cells)", fontsize=20, fontweight='bold', y=0.98)

for i, col in enumerate(target_names):
    sns.scatterplot(x=df_test_real[col], y=df_pred_real[col], ax=axes[i], alpha=0.5, color='#2c7fb8', s=40)
>>>>>>> Stashed changes
    
# 4. Feature importances par target
feature_names = X.columns.tolist()
target_names = y.columns.tolist()

importances_per_target = pd.DataFrame(
    [est.feature_importances_ for est in multi_model.estimators_],
    index=target_names,
    columns=feature_names
)

<<<<<<< Updated upstream
print("\nImportance des features par target :")
print(importances_per_target.round(3))

# 5. Importance moyenne sur les 4 targets
mean_importance = importances_per_target.mean(axis=0).sort_values(ascending=False)

print("\nImportance moyenne (triée) :")
print(mean_importance.round(3))
=======
plt.show()
"""
# Récupération des importances par target
feature_names = X_train.columns.tolist()

importances_per_target = pd.DataFrame(
    [est.feature_importances_ for est in final_model.estimators_],
    index=['Voc', 'Jsc', 'FF', 'PCE'],
    columns=feature_names
)
# transpose pour avoir features en lignes
importances_per_target = importances_per_target.T

# ajouter la moyenne
importances_per_target["Mean"] = importances_per_target.mean(axis=1)
importances_per_target = importances_per_target.sort_values(by="Mean", ascending=False)

importances_per_target.to_csv("importance_target15.csv", index=True, sep=';')

print(importances_per_target.round(3))
>>>>>>> Stashed changes
