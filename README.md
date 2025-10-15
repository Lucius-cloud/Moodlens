# MoodLens — Sentiment Analyzer (happy / sad / neutral)

## Setup (VS Code)

1. Create a folder `MoodLens` and add the files:
   - sample_data.csv
   - train.py
   - predict.py
   - requirements.txt

2. Open folder in VS Code.

3. Create and activate a Python virtual environment:
   - Windows:
     python -m venv venv
     .\venv\Scripts\activate
   - macOS / Linux:
     python3 -m venv venv
     source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

5. Train the model:
   python train.py
   - This will create `moodlens_model.joblib`.

6. Test prediction:
   python predict.py "i feel amazing and happy today"

## Quick Git commands
git init
git add .
git commit -m "Add MoodLens sentiment analyzer"
# Create repo on GitHub and follow instructions, then:
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
