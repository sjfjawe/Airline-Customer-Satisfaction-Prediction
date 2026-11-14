# Airline Customer Satisfaction Prediction
# Predict whether a passenger is satisfied or neutral/dissatisfied
# Data source: https://huggingface.co/datasets/drukeroni/airline-satisfaction-analysis

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(train_path=None, test_path=None):
    """Load train and test datasets from local path or HuggingFace."""
    if train_path and test_path:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    else:
        base_url = "hf://datasets/drukeroni/airline-satisfaction-analysis/"
        df_train = pd.read_csv(base_url + "train.csv")
        df_test = pd.read_csv(base_url + "test.csv")
    return df_train, df_test


def preprocess(df):
    """Clean and engineer features."""
    df = df.copy()
    df.drop(columns=['Unnamed: 0', 'id'], inplace=True, errors='ignore')
    df.columns = df.columns.str.replace(' ', '_')

    # log-transform delay columns
    df['Departure_Delay_in_Minutes'] = np.log1p(df['Departure_Delay_in_Minutes'])
    df['Arrival_Delay_in_Minutes'] = np.log1p(df['Arrival_Delay_in_Minutes'])

    # feature engineering
    df['has_delay'] = ((df['Departure_Delay_in_Minutes'] > 0) | (df['Arrival_Delay_in_Minutes'] > 0)).astype(int)

    # fill missing values
    for col in df.select_dtypes(exclude='object').columns:
        df[col] = df[col].fillna(0)

    return df


def train_and_evaluate():
    """Main training pipeline."""
    print("LOADING DATA...")
    df_train, df_test = load_data()
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    print("PREPROCESSING...")
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    # extract target
    y_full = (df_train['satisfaction'] == 'satisfied').astype(int).values
    y_test = (df_test['satisfaction'] == 'satisfied').astype(int).values
    del df_train['satisfaction']
    del df_test['satisfaction']

    # split train into train/val
    df_train, df_val, y_train, y_val = train_test_split(df_train, y_full, test_size=0.2, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # scale numerical features
    print("SCALING FEATURES...")
    num_features = ['Age', 'Flight_Distance', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']
    scaler = StandardScaler()
    df_train[num_features] = scaler.fit_transform(df_train[num_features])
    df_val[num_features] = scaler.transform(df_val[num_features])
    df_test[num_features] = scaler.transform(df_test[num_features])

    # vectorize
    print("VECTORIZING...")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))
    X_val = dv.transform(df_val.to_dict(orient='records'))
    X_test = dv.transform(df_test.to_dict(orient='records'))

    print(f"Feature count: {len(dv.get_feature_names_out())}")

    # train on validation
    print("\nTRAINING ON VALIDATION...")
    rf = RandomForestClassifier(n_estimators=130, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    val_probs = rf.predict_proba(X_val)[:, 1]
    print(f"Val Accuracy: {accuracy_score(y_val, rf.predict(X_val)):.4f}")
    print(f"Val ROC-AUC:  {roc_auc_score(y_val, val_probs):.4f}")

    # full training
    print("\nFULL TRAINING...")
    df_full = pd.concat([df_train, df_val], ignore_index=True)
    y_full_combined = np.concatenate([y_train, y_val])
    X_full = dv.transform(df_full.to_dict(orient='records'))

    rf = RandomForestClassifier(n_estimators=130, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_full, y_full_combined)

    # test evaluation
    test_probs = rf.predict_proba(X_test)[:, 1]
    print(f"Test Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, test_probs):.4f}")

    # save model
    with open('model.pkl', 'wb') as f_out:
        pickle.dump((rf, dv, scaler), f_out)

    print("\nModel saved to model.pkl")


if __name__ == "__main__":
    train_and_evaluate()
