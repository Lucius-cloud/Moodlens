# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
import os

# If running first time, download punkt (for NLTK; safe small download)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

DATA_PATH = "sample_data.csv"
MODEL_PATH = "moodlens_model.joblib"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text','label'])
    return df['text'].astype(str), df['label'].astype(str)

def build_and_train(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ('nb', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please ensure sample_data.csv exists.")
        return

    X, y = load_data()
    # Ensure stratified split is possible: the test set must contain at least one
    # sample for each class when using `stratify=`. If the dataset is too small
    # for the requested `test_size`, adjust the fraction or fall back to a
    # non-stratified split with a warning.
    requested_test_size = 0.2
    n_samples = len(y)
    class_counts = y.value_counts()
    n_classes = class_counts.shape[0]

    # Minimum number of test samples needed to have at least one per class
    min_test_samples = n_classes

    # Compute concrete test set size (number of samples) from requested fraction
    test_samples = int(requested_test_size * n_samples)

    stratify_arg = y
    if test_samples < min_test_samples:
        # Try to increase test fraction so we have at least one sample per class
        # If even 50% of data is too small (very small dataset), we disable stratify
        # because stratified splitting would be impossible.
        # Compute a new fraction that yields at least min_test_samples (ceiled)
        if n_samples > 0:
            adjusted_fraction = min(0.5, float(min_test_samples) / n_samples)
        else:
            adjusted_fraction = requested_test_size

        new_test_samples = int(adjusted_fraction * n_samples)
        if new_test_samples >= min_test_samples:
            print(f"Adjusted test_size from {requested_test_size} to {adjusted_fraction:.3f} to allow stratified split ({new_test_samples} test samples for {n_classes} classes).")
            requested_test_size = adjusted_fraction
        else:
            print(f"Warning: dataset too small to create a stratified test set with at least one sample per class (classes={n_classes}, samples={n_samples}). Proceeding without stratify.")
            stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=requested_test_size, random_state=42, stratify=stratify_arg
    )

    model = build_and_train(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
