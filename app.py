# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained MoodLens model
MODEL_PATH = "moodlens_model.joblib"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    user_text = ""
    if request.method == "POST":
        user_text = request.form.get("text_input", "")
        if user_text.strip():
            label = model.predict([user_text])[0]
            prediction = label.capitalize()
    return render_template("index.html", prediction=prediction, user_text=user_text)

if __name__ == "__main__":
    # Run on localhost:8000 so port 5000 conflicts are avoided
    app.run(debug=True, host="0.0.0.0", port=8000)
