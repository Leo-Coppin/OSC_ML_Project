import pandas as pd

from sklearn.model_selection import train_test_split
import Compatibility_score_functions as csf

#importing data
X10 = pd.read_csv("Data_Compatibility_score10.csv", sep=';')
y10 = pd.read_csv("Output_Compatibility_score10.csv", sep=';')
target_cols = ['scaled_Voc', 'scaled_Jsc', 'scaled_FF', 'scaled_PCE']
selected_features10 = ["scaled_λ_A_absorption","scaled_HOMO_D","scaled_LUMO_A","scaled_HOMO_A","scaled_EgA_opt","scaled_λ_D_absorption","scaled_EgCV_A","scaled_LUMO_D","scaled_EgCV_D"]
X10 = X10[selected_features10]

#importing data
X15 = pd.read_csv("Data_Compatibility_score15.csv", sep=';')
y15 = pd.read_csv("Output_Compatibility_score15.csv", sep=';')

selected_features15 = ["scaled_delta_lambda", "scaled_λ_A_absorption", "scaled_HOMO_D_LUMO_A", "scaled_HOMO_D", "scaled_HOMO_A", "scaled_delta_HOMO"]
X15 = X15[selected_features15]
i=1
pairs = [[X10, y10], [X15, y15]]
for pair in pairs:
    X, y = pair

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
    study = csf.run_optuna_search(X_train, y_train, input_dim, n_trials=100)

    #Training final model with Optuna Search params
    print("\nTraining final model\n")
    final_model, final_hyperparams = csf.train_final_model(study.best_trial.params, X_train, y_train, input_dim, 500)
    final_model.summary()

    #Evaluation on Final Test set
    #print("\nModel Evaluation\n")
    results, y_pred, y_true = csf.evaluate_model(final_model, X_test, y_test,target_cols)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))

    with open(f"model{i}_parameters.txt", "w") as f:
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
    print("\nModel saved")
    i+=1
    
    
    # one of best models 
    """best_params = {
        'embedding_dim' : 64,
        'n_layers'      : 3,
        'activation'    : 'relu',
        'lr'            : 0.0013622778618182295,
        'optimizer'     : 'adam',
        'weight_decay'  : 0.000572749459793566,
        'dropout_rate'  : 0.06628799569948823,
        'use_batch_norm': True,
        'batch_size'    : 128,
        'patience'      : 6
    }"""