import requests
import pickle
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# API Endpoint
API_URL = "http://130.225.39.127:8000/new_penguin/"

# Paths
MODEL_PATH = "penguin_classifier.pkl"
JSON_OUTPUT_PATH = "latest_prediction.json"
HTML_OUTPUT_PATH = "docs/index.html"

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
    
    df = pd.DataFrame([penguin_data])[features]

    # Standardize numerical features (Ensure consistency with training)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled

def make_prediction(model, encoder, penguin_data):
    """Predict the species of the new penguin."""
    X_new = preprocess_new_data(penguin_data)
    pred_class = model.predict(X_new)[0]
    pred_species = encoder.inverse_transform([pred_class])[0]
    
    return pred_species

def save_prediction(penguin_data, predicted_species):
    """Save prediction result to JSON and update HTML page for GitHub Pages."""
    result = penguin_data.copy()
    result["predicted_species"] = predicted_species
    result["timestamp"] = datetime.utcnow().isoformat()

    # Save JSON
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"Prediction saved to {JSON_OUTPUT_PATH}")

    # Update HTML
    update_html(predicted_species, result["timestamp"])

def update_html(species, timestamp):
    """Update docs/index.html with the latest prediction."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Daily Penguin Prediction</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 20px; }}
            .container {{ max-width: 600px; margin: auto; background: #f9f9f9; padding: 20px; border-radius: 10px; }}
            h1 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Latest Penguin Prediction</h1>
            <p><strong>Predicted Species:</strong> {species}</p>
            <p><strong>Prediction Time:</strong> {timestamp}</p>
        </div>
    </body>
    </html>
    """
    with open(HTML_OUTPUT_PATH, "w") as f:
        f.write(html_content)

    print(f"Updated {HTML_OUTPUT_PATH}")

if __name__ == "__main__":
    model, encoder = load_model()
    penguin_data = fetch_new_penguin()

    if penguin_data:
        predicted_species = make_prediction(model, encoder, penguin_data)
        save_prediction(penguin_data, predicted_species)
        print(f"Predicted Species: {predicted_species}")
