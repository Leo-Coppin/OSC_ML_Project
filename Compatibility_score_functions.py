import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, BatchNormalization, Dropout, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam, AdamW
from keras.metrics import R2Score

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

def build_model(input_dim, hyperparams):
    #input layer
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    #add i layers with unit neurons
    for i, units in enumerate(hyperparams['hidden_layers']):
        x = Dense(units)(x)
        
        #add batch normalization if True
        if hyperparams['use_batch_norm']:
            x = BatchNormalization()(x)
        
        #add activation function to layer
        x = Activation(hyperparams['activation'])(x)
        
        if hyperparams['dropout_rate']>0:
            x = Dropout(hyperparams['dropout_rate'])(x)
    
    #last layer -> embedding layer
    embedding = Dense(
        hyperparams['embedding_dim'],
        activation=hyperparams['activation'],
        name = 'embedding_CS'
    )(x)
    
    #output layers -> 4 neurons = Jsc, Voc, FF, PCE
    outputs = Dense(4, name='output', activation='linear')(embedding)
    
    model = Model(inputs = inputs, outputs=outputs, name='ANN_CS_OSC')
    
    #select optimizer
    if hyperparams['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=hyperparams['lr'])
    elif hyperparams['optimizer'] == 'adamw':
        optimizer = AdamW(
            learning_rate=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
        
    #compile model -> loss and metrics is mean squared error
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae','mse', R2Score(num_regressors=0, name='r2')]
    )
    
    return model


#training the previous model -> return training history
def train_model(model, X_train, y_train, X_val, y_val, hyperparams, extra_callbacks=[]):
    #stop if no amelioration and restore best weights
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=hyperparams['patience'],
        restore_best_weights=True            
    )
 
    #reduce lr
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,          
        patience=10,
        min_lr=1e-6
    )
    all_callbacks = [early_stop, reduce_lr]
    if extra_callbacks != []:
        all_callbacks.append(extra_callbacks)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        callbacks=all_callbacks,
        verbose=0
    )
 
    return history


def evaluate_model(model, X_test, y_test, target_cols):
    """
    Évalue le modèle sur le test set.
    Dénormalise les prédictions pour avoir les métriques dans les unités réelles.
    """
    y_pred = model.predict(X_test, verbose=0)
 
    results = {}
    for i, target in enumerate(target_cols):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        results[target] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
 
    return results, y_pred, y_test



def create_optuna_objective(X_train_val, y_train_val, input_dim, n_folds=10):
    #Create objective function for Optuna
    
    # Each Trial test a combination of hyperparams with K-FoldCV  
    def objective(trial):
        # Hyperparams search
        # size of embedding layers
        embedding_dim = trial.suggest_categorical('embedding_dim', [16, 32, 64])
        
        # number of hidden layers
        n_layers = trial.suggest_int('n_layers', 2, 5)

        # size of each layers -> decreasing into funnel shape -> decrease by power of 2
        hidden_layers = [
            min(embedding_dim * (2 ** (n_layers - i)), 256)
            for i in range(n_layers)
        ]
        
        #dictionnary containing hyperparams
        hyperparams = {
            'hidden_layers'  : hidden_layers,
            'embedding_dim'  : embedding_dim,
            'activation'     : trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu']),
            'lr'             : trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'optimizer'      : trial.suggest_categorical('optimizer', ['adam', 'adamw']),
            'weight_decay'   : trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'dropout_rate'   : trial.suggest_float('dropout_rate', 0.0, 0.3),
            'use_batch_norm' : trial.suggest_categorical('use_batch_norm', [True, False]),
            'batch_size'     : trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs'         : 500,     # early stopping s'en chargera
            'patience'       : trial.suggest_int('patience', 5, 40),
        }
 
        # --- K-Fold Cross Validation ---
        X_train_val_np = np.array(X_train_val)
        y_train_val_np = np.array(y_train_val)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        val_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val_np)):
            X_fold_train = X_train_val_np[train_idx]   # ✓ indexation NumPy
            X_fold_val   = X_train_val_np[val_idx]
            y_fold_train = y_train_val_np[train_idx]
            y_fold_val   = y_train_val_np[val_idx]
 
            model = build_model(input_dim, hyperparams)
            history = train_model(
                model, X_fold_train, y_fold_train,
                X_fold_val, y_fold_val, hyperparams
            )
 
            # Score = best val_loss for this fold
            best_val_loss = min(history.history['val_loss'])
            val_losses.append(best_val_loss)
 
            # Pruning Optuna : stop bad trials early -> optimize time and ressources
            trial.report(np.mean(val_losses), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
 
        return np.mean(val_losses)   # Optuna minimize this value
 
    return objective
 
 
def run_optuna_search(X_train_val, y_train_val, input_dim, n_trials=150):
    # starts Optuna Search
    # n_trials=50 good place to start
    
    # Pruner : stops bad trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
 
    study = optuna.create_study(
        direction='minimize',      # minimize val_loss
        pruner=pruner,
        study_name='ANN_CS_OSC_hyperopt'
    )
 
    objective = create_optuna_objective(X_train_val, y_train_val, input_dim)
 
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )
 
    print(f"\nMeilleur trial : {study.best_trial.number}")
    print(f"Meilleure val_loss : {study.best_trial.value:.4f}")
    print(f"Meilleurs hyperparamètres :\n{study.best_trial.params}")
 
    return study


#training with best hyperparameters found
def train_final_model(best_params, X_train_val, y_train_val, input_dim, epoch):
    # Reconstruction model based on best Optuna parameters
    n_layers = best_params['n_layers']
    embedding_dim = best_params['embedding_dim']

    hidden_layers = [
        min(embedding_dim * (2 ** (n_layers - i)), 256)
        for i in range(n_layers)
    ]

    hyperparams = {
        'hidden_layers'  : hidden_layers,
        'embedding_dim'  : embedding_dim,
        'activation'     : best_params['activation'],
        'lr'             : best_params['lr'],
        'optimizer'      : best_params['optimizer'],
        'weight_decay'   : best_params.get('weight_decay', 1e-4),
        'dropout_rate'   : best_params['dropout_rate'],
        'use_batch_norm' : best_params['use_batch_norm'],
        'batch_size'     : best_params['batch_size'],
        'epochs'         : epoch,
        'patience'       : best_params['patience'],
    }

    X_train_val_np = np.array(X_train_val)
    y_train_val_np = np.array(y_train_val)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_val_np, y_train_val_np, test_size=0.1, random_state=42
    )

    model = build_model(input_dim, hyperparams)
    
    # --- Callback TensorBoard ---
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir='./tensorboard_logs',   # dossier où les logs sont sauvegardés
        histogram_freq=1,               # histogramme des poids à chaque epoch
        write_graph=True,
    )
    
    train_model(model, X_tr, y_tr, X_val, y_val, hyperparams, extra_callbacks=tensorboard_cb)

    return model, hyperparams