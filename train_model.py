# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_FILE = 'strength_data.csv'
MODEL_FILE = 'strength_model.pkl'
SCALER_FILE = 'scaler.pkl'
TEST_SIZE = 0.2
RANDOM_SEED = 42

def train_and_save_model():
    """
    Loads data, preprocesses it (assuming weights are in KG), 
    trains a RandomForestClassifier, and saves the model and the scaler objects.
    """
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please ensure the CSV file is created.")
        return

    print(f"Loading data from {DATA_FILE}...")
    try:
        # 2. Load data
        df = pd.read_csv(DATA_FILE)
        # Weights (bench, squat, deadlift) are assumed to be in Kilograms (KG).
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    # 3. Features and target
    # New feature set: bench, squat, deadlift, gender, pull ups, push ups
    X = df.drop('strong', axis=1)
    y = df['strong']  # 1 = strong, 0 = not strong

    # 4. Encode gender (assuming 'male' and 'female' strings)
    print("Encoding gender feature...")
    gender_mapping = {'male': 0, 'female': 1}
    X['gender'] = X['gender'].map(gender_mapping)

    # 5. Scale numeric features (all features will be scaled together)
    scaler = StandardScaler()
    print("Fitting and transforming all features...")
    X_scaled = scaler.fit_transform(X)
    
    # 6. 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")

    # 7. Train the model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    print("Training complete.")

    # 8. Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy on test set: {accuracy:.2f}")

    # 9. Save the trained model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")

if __name__ == '__main__':
    train_and_save_model()
