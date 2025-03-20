import requests
import pickle
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# API Endpoint
API_URL = "http://130.225.39.127:8000/new_penguin/"

# Load the trained model and label encoder
MODEL_PATH = "penguin_classifier.pkl"

def load_model():
    """Load the trained model and label encoder."""
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["encoder"]

def fetch_new_penguin():
    """Fetch new penguin data from the API."""
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code}")
        return None

def preprocess_new_data(penguin_data):
    """Preprocess new data for model prediction."""
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    # Convert JSON to DataFrame
    df = pd.DataFrame([penguin_data])[features]

    # Standardize numerical features (Ensure it's the same as used in training)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # 🔥 Ensure consistency with training

    return df_scaled

def make_prediction(model, encoder, penguin_data):
    """Predict the species of the new penguin."""
    X_new = preprocess_new_data(penguin_data)
    pred_class = model.predict(X_new)[0]
    pred_species = encoder.inverse_transform([pred_class])[0]
    
    return pred_species

def save_prediction(penguin_data, predicted_species):
    """Save the prediction result to a JSON file, ensuring updates for GitHub Actions."""
    result = penguin_data.copy()
    result["predicted_species"] = predicted_species
    result["timestamp"] = datetime.utcnow().isoformat()  # 🔥 Ensures file change

    output_path = "latest_prediction.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    # Load the trained model
    model, encoder = load_model()

    # Fetch new penguin data
    penguin_data = fetch_new_penguin()

    if penguin_data:
        # Make prediction
        predicted_species = make_prediction(model, encoder, penguin_data)

        # Save the result
        save_prediction(penguin_data, predicted_species)

        print(f"Predicted Species: {predicted_species}")
