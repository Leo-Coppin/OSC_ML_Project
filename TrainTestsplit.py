'''from sklearn.model_selection import GroupShuffleSplit

def validation(df, test_size=0.3, random_state=42,
               donor_col="SMILES_don", acceptor_col="SMILES_acc",
               target_cols=None):

    df = df.copy()

    if target_cols is None:
        target_cols = ['Voc', 'Jsc', 'FF', 'PCE', 'delta_HOMO', 'delta_LUMO']

    required_cols = [donor_col, acceptor_col] + target_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Separate X and y
    X = df.drop(columns=target_cols).copy()
    y = df[target_cols].copy()

    #Group by "SMILES_don"
    groups = df[donor_col].astype(str)

    #Apply the GroupShuffleSplit to split into 70/30 and ensure that no same donor groups are in both the train and test data
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    #Creating x_train, y_train and x_test,y_test subsets from the train_idx and test_idx
    x_train = X.iloc[train_idx].copy()
    x_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    print("Validation: Grouped split done.")
    print(f"Train size: {len(x_train)}")
    print(f"Test size: {len(x_test)}")
    
    
    return x_train, x_test, y_train, y_test
'''