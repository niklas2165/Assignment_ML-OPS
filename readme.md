# Penguins of Madagascar - MLOps Assignment

## Approach

### Task 1: GitHub Repository & Documentation
- **Repository Setup:** A public GitHub repository was created with a detailed README describing the project.
- **Documentation:** The README outlines the project structure, approach, and usage instructions.

### Task 2: Data Transformation & Database Creation
- **File:** `data_to_db.py`
- **Approach:** 
  - Load the penguins dataset from Seaborn.
  - Clean the dataset (dropping rows with missing values).
  - Transform the data by mapping island names to unique IDs and generating a unique `animal_id` for each penguin.
  - Insert the processed data into two tables (PENGUINS and ISLANDS) in an SQLite database (`penguins.db`).

### Task 3: Feature Selection & Model Training
- **File:** `feature_selection.py`
- **Approach:**
  - Load data from the database.
  - Encode categorical variables and standardize numerical features.
  - Apply several feature selection methods (Filter, Wrapper, Embedded, and Permutation Importance) to determine the most relevant features.
  - Based on the results, select the features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g` (dropping `sex`).

- **File:** `train_model.py`
- **Approach:**
  - Load the selected features from the database.
  - Preprocess the data by encoding the target variable and scaling numerical features.
  - Split the data into training and testing sets.
  - Train a RandomForest classifier.
  - Evaluate model performance (achieving 100% accuracy on the test set, though further investigation may be needed for potential overfitting or leakage).
  - Save the trained model and label encoder to `penguin_classifier.pkl`.

### Task 4: Automated Daily Predictions & GitHub Pages
- **File:** `daily_prediction.py`
- **Approach:**
  - Fetch new penguin data every day from the provided API endpoint.
  - Load the saved trained model (`penguin_classifier.pkl`).
  - Preprocess the incoming data and make predictions.
  - Save the prediction results to `latest_prediction.json`.
  - **Update `docs/index.html`** so that GitHub Pages always displays the latest prediction.

- **File:** `.github/workflows/predict.yml`
- **Approach:**
  - A GitHub Actions workflow is set up to run `daily_prediction.py` every day at 7:30 AM UTC.
  - The workflow commits and pushes the updated prediction file (`latest_prediction.json`) and `docs/index.html` to the repository.
  - This ensures GitHub Pages **displays the latest prediction**.

- **GitHub Pages Setup:**
  - The project **hosts a public webpage** displaying the latest penguin prediction at:  
    👉 **[https://niklas2165.github.io/Assignment_ML-OPS/](https://niklas2165.github.io/Assignment_ML-OPS/)**
  - The webpage automatically updates daily.

## Usage Instructions
1. **Data Preparation:**  
   Run `data_to_db.py` to create and populate the SQLite database (`penguins.db`).

2. **Feature Selection:**  
   Run `feature_selection.py` to view the feature importance and determine the best features to use.

3. **Model Training:**  
   Run `train_model.py` to train the classifier and save the model to `penguin_classifier.pkl`.

4. **Daily Predictions & Web Updates:**  
   - The `daily_prediction.py` script will be triggered automatically by the GitHub Actions workflow (`.github/workflows/predict.yml`) every day at 7:30 AM UTC.  
   - The **latest prediction** is displayed on GitHub Pages at:  
     👉 **[https://niklas2165.github.io/Assignment_ML-OPS/](https://niklas2165.github.io/Assignment_ML-OPS/)**  

## Final Notes
- **Model Accuracy:**  
  The model achieved 100% accuracy on the test set. While promising, further checks (e.g., cross-validation) are recommended to rule out overfitting or data leakage.

- **Automation & Deployment:**  
  - The project **automatically fetches new data, makes predictions, and updates a live webpage**.  
  - Predictions are stored in `latest_prediction.json`, and GitHub Pages serves the results via `docs/index.html`.  
  - Everything runs **without manual intervention** through GitHub Actions.
