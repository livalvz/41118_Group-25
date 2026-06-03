# train_classifier.py
 
import os
import sys
import joblib
import pandas as pd
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
 
 
DATA_PATH = "gesture_landmarks.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_classifier.pkl")
 
# Only train the static hand-shape classifier on these labels.
# Dynamic gestures are detected separately in gesture_recognition_node.py.
ALLOWED_LABELS = [
    "one_finger",
    "two_fingers",
    "three_fingers",
    "open_palm",
    "closed_fist",
]
 
 
def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Could not find {DATA_PATH}")
        print("Run collect_landmark.py first to collect gesture samples.")
        sys.exit(1)
 
    data = pd.read_csv(DATA_PATH)
 
    if "label" not in data.columns:
        print("CSV file must contain a 'label' column.")
        sys.exit(1)
 
    # Remove old dynamic/static direction labels from older datasets.
    original_count = len(data)
    data = data[data["label"].isin(ALLOWED_LABELS)]
    filtered_count = len(data)
 
    removed_count = original_count - filtered_count
 
    if removed_count > 0:
        print(f"Filtered out {removed_count} old samples not used by the static classifier.")
 
    if len(data) < 10:
        print("Not enough static gesture data yet. Collect more samples first.")
        sys.exit(1)
 
    X = data.drop("label", axis=1)
    y = data["label"]
 
    return X, y
 
 
def print_dataset_summary(y):
    print("\nDataset summary:")
    print("----------------")
 
    counts = y.value_counts()
 
    for label, count in counts.items():
        print(f"{label}: {count} samples")
 
    print(f"\nTotal samples: {len(y)}")
    print(f"Total classes: {len(counts)}")
 
 
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
 
    model.fit(X_train, y_train)
 
    return model
 
 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
 
    accuracy = accuracy_score(y_test, y_pred)
 
    print("\nModel evaluation:")
    print("-----------------")
    print(f"Accuracy: {accuracy:.3f}")
 
    print("\nClass order:")
    print(model.classes_)
 
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
 
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
 
 
def save_model(model, feature_names, class_names):
    os.makedirs(MODEL_DIR, exist_ok=True)
 
    model_data = {
        "model": model,
        "feature_names": feature_names,
        "class_names": class_names,
    }
 
    joblib.dump(model_data, MODEL_PATH)
 
    print(f"\nSaved model to: {MODEL_PATH}")
 
 
def main():
    X, y = load_data()
 
    print_dataset_summary(y)
 
    class_counts = y.value_counts()
 
    if len(class_counts) < 2:
        print("\nYou need at least 2 different static gesture classes to train a classifier.")
        sys.exit(1)
 
    if class_counts.min() < 2:
        print("\nEach static gesture needs at least 2 samples before training.")
        print("Collect more samples for the smallest class.")
        sys.exit(1)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
 
    print("\nTraining model...")
    model = train_model(X_train, y_train)
 
    evaluate_model(model, X_test, y_test)
 
    save_model(
        model,
        feature_names=list(X.columns),
        class_names=sorted(y.unique().tolist())
    )
 
 
if __name__ == "__main__":
    main()