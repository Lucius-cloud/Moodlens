# predict.py
import joblib
import sys
import argparse

MODEL_PATH = "moodlens_model.joblib"

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict_text(model, text):
    pred = model.predict([text])[0]
    proba = None
    try:
        proba = model.predict_proba([text])[0]
    except Exception:
        pass
    return pred, proba

def main():
    parser = argparse.ArgumentParser(description="MoodLens: Predict sentiment (happy/sad/neutral)")
    parser.add_argument('text', nargs='+', help='Text to analyze')
    args = parser.parse_args()
    text = " ".join(args.text)

    model = load_model()
    label, proba = predict_text(model, text)
    print(f"Input: {text}")
    print(f"Predicted label: {label}")
    if proba is not None:
        print("Probabilities:", proba)

if __name__ == "__main__":
    main()
