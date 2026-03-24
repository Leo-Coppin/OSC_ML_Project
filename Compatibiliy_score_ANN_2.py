import pandas as pd

from sklearn.model_selection import train_test_split

import Compatibility_score_functions as csf

#importing data
X = pd.read_csv("Data_Compatibility_score.csv", sep=';')
y = pd.read_csv("Output_Compatibility_score.csv", sep=';')
target_cols = ['scaled_Voc', 'scaled_Jsc', 'scaled_FF', 'scaled_PCE']


#spliting datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#conversion in numpy values
X_train = X_train.values
X_test  = X_test.values
y_train = y_train.values
y_test  = y_test.values

input_dim = X_train.shape[1]


#Optuna Search
print("\nOptuna Search\n")
study = csf.run_optuna_search(X_train, y_train, input_dim, n_trials=50)

""" one of best models 
best_params = {
    'embedding_dim' : 64,
    'n_layers'      : 3,
    'activation'    : 'relu',
    'lr'            : 0.0013622778618182295,
    'optimizer'     : 'adam',
    'weight_decay'  : 0.000572749459793566,
    'dropout_rate'  : 0.06628799569948823,
    'use_batch_norm': True,
    'batch_size'    : 128,
}
# epoch = 500 
#                MAE    RMSE      R2
# scaled_Voc  0.4041  0.6657  0.5241
# scaled_Jsc  0.3870  0.5387  0.7087
# scaled_FF   0.5689  0.8194  0.4170
# scaled_PCE  0.4216  0.5627  0.6741
"""

#Training final model with Optuna Search params
print("\nTraining final model\n")
final_model, final_hyperparams = csf.train_final_model(study.best_trial.params, X_train, y_train, input_dim)
final_model.summary()


#Evaluation on Final Test set
print("\nModel Evaluation\n")
results, y_pred, y_true = csf.evaluate_model(final_model, X_test, y_test,target_cols)

results_df = pd.DataFrame(results).T
print(results_df.round(4))

final_model.save('ann_CS_OSC.keras')
print("\nModel saved")