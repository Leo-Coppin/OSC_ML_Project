import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

y_train = pd.read_csv("train_dataset.csv", sep=';')
y_test = pd.read_csv("test_dataset.csv", sep=';')

outputs = ["Voc", "Jsc", "FF", "PCE", "delta_HOMO", "delta_LUMO"]

y_train = y_train[outputs]
y_test = y_test[outputs]

X_train = pd.read_csv("Data_Morgan_train.csv", sep=";")
X_test = pd.read_csv("Data_Morgan_test.csv", sep=";")

rf = RandomForestRegressor(random_state=42, n_estimators=500)
multi_rf = MultiOutputRegressor(rf)

multi_rf.fit(X_train, y_train)
y_pred = multi_rf.predict(X_test)

r2_per_col = r2_score(y_test, y_pred, multioutput='raw_values')
mae_per_col = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
mse_per_col = mean_squared_error(y_test, y_pred, multioutput='raw_values')

results_detailed = pd.DataFrame({
    'R²': r2_per_col,
    'MAE': mae_per_col,
    'MSE': mse_per_col
}, index=outputs)

print(f"\nDetails per property for training dataset :")
print(results_detailed.round(4))


# evaluation over all shuffles 
for i in range(1,8):
    X_test = pd.read_csv(f"DataShuffle/Data_Morgan_test_{i}.csv", sep=";")
    y_test = pd.read_csv(f"DataShuffle/test_dataset_{i}.csv", sep=';')
    y_test = y_test[outputs]

    y_pred = multi_rf.predict(X_test)

    r2_per_col = r2_score(y_test, y_pred, multioutput='raw_values')
    mae_per_col = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse_per_col = mean_squared_error(y_test, y_pred, multioutput='raw_values')

    results_detailed = pd.DataFrame({
        'R²': r2_per_col,
        'MAE': mae_per_col,
        'MSE': mse_per_col
    }, index=outputs)

    print(f"\nDetails per property for shuffle : {i}")
    print(results_detailed.round(4))
  
