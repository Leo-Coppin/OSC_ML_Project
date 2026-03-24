import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error


X = pd.read_csv("Data_Compatibility_score.csv", sep=';')
y = pd.read_csv("Output_Compatibility_score.csv", sep=';')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 300,
    'subsample': 0.8,
    'random_state': 42
}

base_gbm = GradientBoostingRegressor(**best_params)
final_model = MultiOutputRegressor(base_gbm) 

# I used grid search to find the best hyperparameters for the Gradient Boosting Regressor, but it is commented out to save time during execution. The best parameters found are used to initialize the base_gbm model.
# param_grid = {
#     'estimator__n_estimators': [100, 200, 300],
#     'estimator__learning_rate': [0.01, 0.05, 0.1],
#     'estimator__max_depth': [3, 4, 5],
#     'estimator__subsample': [0.8, 1.0]
# }

# grid_search = GridSearchCV(
#     multi_model, 
#     param_grid, 
#     cv=5, # 5-fold cross-validation
#     scoring='r2', # R² score as the evaluation metric, R² for FF is only around 0.29 wereas for the other three it is around 0.5, so it is harder to determine what impact the most the FF factor.
#     n_jobs=-1 # Use all available CPU cores for parallel processing
# )

# grid_search.fit(X_train, y_train)

# final_model = grid_search.best_estimator_

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

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

plt.close('all') 

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

fig.suptitle("Experimental vs. Predicted Values (Organic Solar Cells)", fontsize=20, fontweight='bold', y=0.98)

for i, col in enumerate(target_names):
    sns.scatterplot(x=df_test_real[col], y=df_pred_real[col], ax=axes[i], alpha=0.5, color='#2c7fb8', s=40)
    
    axis_min = min(df_test_real[col].min(), df_pred_real[col].min())
    axis_max = max(df_test_real[col].max(), df_pred_real[col].max())
    axes[i].plot([axis_min, axis_max], [axis_min, axis_max], 'r--', lw=2)
    
    axes[i].set_title(f"Parity Plot: {col}", fontsize=15, fontweight='bold')
    axes[i].set_xlabel("Experimental Value", fontsize=12)
    axes[i].set_ylabel("Predicted Value", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.3, hspace=0.4)

plt.show()