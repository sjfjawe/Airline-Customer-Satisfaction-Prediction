# Airline Customer Satisfaction Prediction - Inference
# Loads trained model and runs predictions on new data

import pickle
import numpy as np
import pandas as pd


def load_model(model_path='model.pkl'):
    """Load trained model, vectorizer, and scaler."""
    with open(model_path, 'rb') as f_in:
        model, dv, scaler = pickle.load(f_in)
    return model, dv, scaler


def preprocess(df, scaler):
    """Preprocess input data for prediction."""
    df = df.copy()
    df.drop(columns=['Unnamed: 0', 'id'], inplace=True, errors='ignore')
    df.columns = df.columns.str.replace(' ', '_')

    # Set defaults for optional fields
    if 'Flight_Distance' not in df.columns:
        df['Flight_Distance'] = 0
    if 'Departure_Delay_in_Minutes' not in df.columns:
        df['Departure_Delay_in_Minutes'] = 0
    if 'Arrival_Delay_in_Minutes' not in df.columns:
        df['Arrival_Delay_in_Minutes'] = 0

    # log-transform delay columns
    df['Departure_Delay_in_Minutes'] = np.log1p(df['Departure_Delay_in_Minutes'])
    df['Arrival_Delay_in_Minutes'] = np.log1p(df['Arrival_Delay_in_Minutes'])

    # feature engineering: use has_delay if provided, otherwise compute from delays
    if 'has_delay' not in df.columns:
        df['has_delay'] = ((df['Departure_Delay_in_Minutes'] > 0) | (df['Arrival_Delay_in_Minutes'] > 0)).astype(int)

    # fill missing values
    for col in df.select_dtypes(exclude='object').columns:
        df[col] = df[col].fillna(0)

    # scale numerical features
    num_features = ['Age', 'Flight_Distance', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']
    df[num_features] = scaler.transform(df[num_features])

    return df


def predict(data, model, dv, scaler):
    """
    Run prediction on input data.

    Args:
        data: dict, list of dicts, or DataFrame
        model: trained classifier
        dv: fitted DictVectorizer
        scaler: fitted StandardScaler

    Returns:
        dict with 'prediction' (0/1) and 'probability' (float)
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    df = preprocess(df, scaler)
    X = dv.transform(df.to_dict(orient='records'))

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        'prediction': pred.tolist(),
        'probability': prob.tolist(),
        'label': ['satisfied' if p == 1 else 'neutral or dissatisfied' for p in pred]
    }


if __name__ == "__main__":
    # Example usage
    model, dv, scaler = load_model()

    # Sample passenger data
    sample = {
        'Gender': 'Female',
        'Customer_Type': 'Loyal Customer',
        'Age': 35,
        'Type_of_Travel': 'Business travel',
        'Class': 'Business',
        'Flight_Distance': 1500,
        'Inflight_wifi_service': 4,
        'Departure/Arrival_time_convenient': 4,
        'Ease_of_Online_booking': 4,
        'Gate_location': 3,
        'Food_and_drink': 4,
        'Online_boarding': 5,
        'Seat_comfort': 4,
        'Inflight_entertainment': 5,
        'On-board_service': 4,
        'Leg_room_service': 4,
        'Baggage_handling': 4,
        'Checkin_service': 4,
        'Inflight_service': 4,
        'Cleanliness': 4,
        'Departure_Delay_in_Minutes': 0,
        'Arrival_Delay_in_Minutes': 0
    }

    result = predict(sample, model, dv, scaler)
    print(f"Prediction: {result['label'][0]}")
    print(f"Probability of satisfaction: {result['probability'][0]:.4f}")
