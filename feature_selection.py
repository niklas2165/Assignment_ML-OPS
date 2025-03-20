import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

# Load Data from Database
def load_data_from_db(db_name="penguins.db"):
    """Load the penguin dataset from the SQLite database."""
    conn = sqlite3.connect(db_name)
    query = """
        SELECT species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex 
        FROM PENGUINS;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Load and preprocess the data
df = load_data_from_db()

# Encode categorical variables
label_encoder_species = LabelEncoder()
label_encoder_sex = LabelEncoder()

df["species"] = label_encoder_species.fit_transform(df["species"])  # Target
df["sex"] = label_encoder_sex.fit_transform(df["sex"])  # Encode 'Male' / 'Female' as 0/1

# Define features (X) and target variable (y)
X = df.drop(columns=["species"])  # Features
y = df["species"]  # Target

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature Names
feature_names = X.columns

### 1. Filter Method: ANOVA (f_classif)
selector = SelectKBest(score_func=f_classif, k="all")  # Keep all for evaluation
X_train_selected = selector.fit_transform(X_train, y_train)
anova_scores = selector.scores_

### 2. Wrapper Method: Recursive Feature Elimination (RFE)
logreg = LogisticRegression()
rfe = RFE(estimator=logreg, n_features_to_select=3)
X_train_selected_rfe = rfe.fit_transform(X_train, y_train)
rfe_ranking = rfe.ranking_

### 3. Embedded Method: Feature Importance using RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_importance = rf.feature_importances_

### 4. Permutation Importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_scores = perm_importance.importances_mean

# Combine results into a DataFrame
feature_scores = pd.DataFrame({
    "Feature": feature_names,
    "ANOVA Score": anova_scores,
    "RFE Rank": rfe_ranking,
    "RandomForest Importance": rf_importance,
    "Permutation Importance": perm_scores
})

# Sort by RandomForest Importance
feature_scores = feature_scores.sort_values(by="RandomForest Importance", ascending=False)

# Print the feature selection results
print("\n=== Feature Selection Results ===")
print(feature_scores)

# Plot Feature Importances
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_scores["Feature"], y=feature_scores["RandomForest Importance"], palette="viridis")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance (RandomForest)")
plt.xticks(rotation=45)
plt.show()
