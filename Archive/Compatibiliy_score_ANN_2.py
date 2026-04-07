import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import Compatibility_score_functions as csf
import keras

#importing data
X = pd.read_csv("Data_Compatibility_score.csv", sep=';')
y = pd.read_csv("Output_Compatibility_score.csv", sep=';')

target_cols = ['scaled_Voc', 'scaled_Jsc', 'scaled_FF', 'scaled_PCE']
selected_features = ["scaled_λ_A_absorption","scaled_HOMO_D","scaled_LUMO_A","scaled_HOMO_A","scaled_EgA_opt","scaled_λ_D_absorption","scaled_EgCV_A","scaled_LUMO_D","scaled_EgCV_D"]

X = X[selected_features]

#spliting datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#conversion in numpy values
X_train = X_train.values
X_test  = X_test.values
y_train = y_train.values
y_test  = y_test.values

input_dim = X_train.shape[1]

#Optuna Search
# print("\nOptuna Search\n")
# study = csf.run_optuna_search(X_train, y_train, input_dim, n_trials=100)

#Meilleure val_loss : 0.4226
best_params ={
    'embedding_dim': 64, 
    'n_layers': 3, 
    'activation': 'relu', 
    'lr': 0.0018539272115348931, 
    'optimizer': 'adamw', 
    'weight_decay': 1.4478963830821665e-05, 
    'dropout_rate': 0.023900814972888672, 
    'use_batch_norm': True, 
    'batch_size': 64, 
    'patience': 20
}

best_score     = -np.inf    # on maximise R² moyen
best_model     = None
best_results   = None
best_run       = -1

temp_model_path = "best_ann_CS_OSC_temp.keras"
"""
for run in range (200):
    print(f"run n{run}")
#Training final model with best params
    print(f"\nTraining final model n{run}")
    final_model, final_hyperparams = csf.train_final_model(best_params, X_train, y_train, input_dim, 1000)
    #final_model.summary()

#Evaluation on Final Test set
    print(f"\nModel Evaluation n{run}")
    results, y_pred, y_true = csf.evaluate_model(final_model, X_test, y_test,target_cols)
    results_df = pd.DataFrame(results).T
    #print(results_df.round(4))

    mean_r2   =  results_df['R2'].sum()
    mean_mae  =  results_df['MAE'].sum()
    mean_rmse =  results_df['RMSE'].sum()
    composite_score = mean_r2 - mean_mae - mean_rmse
    
    print(f"R² moyen : {mean_r2:.4f} | MAE moyen : {mean_mae:.4f} | MSE moyen : {mean_rmse:.4f}")
    print(f"Best composite score : {best_score}, current score : {composite_score}")
    
    if composite_score > best_score:
        best_score   = composite_score
        best_results = results_df
        best_run     = run
        final_model.save(temp_model_path)
        print(f"  → Nouveau meilleur modèle (run {best_run})")
"""
best_model = keras.models.load_model(temp_model_path) 
# Meilleur score composite : -1.7910
# Résultats du meilleur modèle :
#                MAE    RMSE      R2
# scaled_Voc  0.4169  0.6440  0.5546
# scaled_Jsc  0.3622  0.5192  0.7294
# scaled_FF   0.5439  0.7840  0.4663
# scaled_PCE  0.4064  0.5516  0.6869

X_RDKit = pd.read_csv("Data_RDKit.csv", sep=";")
y_RDKit = pd.read_csv("Output_RDkit.csv", sep=";")
y_RDKit = csf.extract_compatibility_index(best_model, X_RDKit, y_RDKit, selected_features)
y_RDKit.to_csv("Output_RDKit.csv", index=False, sep=";")

X_Mordred = pd.read_csv("Data_Mordred.csv", sep=";")
y_Mordred = pd.read_csv("Output_Mordred.csv", sep=";")
y_Mordred = csf.extract_compatibility_index(best_model, X_Mordred, y_Mordred, selected_features)
y_Mordred.to_csv("Output_Mordred.csv", index=False, sep=";")

X_Morgan = pd.read_csv("Data_Morgan.csv", sep=";")
y_Morgan = pd.read_csv("Output_Morgan.csv", sep=";")
y_Morgan = csf.extract_compatibility_index(best_model, X_Morgan, y_Morgan, selected_features)
y_Morgan.to_csv("Output_Morgan.csv", index=False, sep=";")

X_MACCS = pd.read_csv("Data_MACCS.csv", sep=";")
y_MACCS = pd.read_csv("Output_MACCS.csv", sep=";")
y_MACCS = csf.extract_compatibility_index(best_model, X_MACCS, y_MACCS, selected_features)
y_MACCS.to_csv("Output_MACCS.csv", index=False, sep=";")

X_PubChem = pd.read_csv("Data_PubChem.csv", sep=";")
y_PubChem = pd.read_csv("Output_PubChem.csv", sep=";")
y_PubChem = csf.extract_compatibility_index(best_model, X_PubChem, y_PubChem, selected_features)
y_PubChem.to_csv("Output_PubChem.csv", index=False, sep=";")