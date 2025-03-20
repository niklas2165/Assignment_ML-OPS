import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def load_data_from_db(db_name="penguins.db"):
    """Load selected features from the SQLite database."""
    conn = sqlite3.connect(db_name)
    query = """
        SELECT species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
        FROM PENGUINS;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    """Preprocess the data: Encode target variable and scale numerical features."""
    
    # Encode target variable (species)
    label_encoder = LabelEncoder()
    df["species"] = label_encoder.fit_transform(df["species"])

    # Features and target variable
    X = df.drop(columns=["species"])  # Features
    y = df["species"]  # Target (classification label)

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, label_encoder  # Return label encoder too

def train_and_save_model(X, y, label_encoder, model_filename="penguin_classifier.pkl"):
    """Train a RandomForest classifier and save the trained model."""
    
    # Split dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print performance metrics
    print("\n=== Model Training Results ===")
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model and label encoder
    with open(model_filename, "wb") as f:
        pickle.dump({"model": model, "encoder": label_encoder}, f)
    
    print(f"\nModel saved as {model_filename}")

if __name__ == "__main__":
    # Load data
    df = load_data_from_db()

    # Preprocess data
    X, y, encoder = preprocess_data(df)

    # Train and save model
    train_and_save_model(X, y, encoder)  # Pass encoder
