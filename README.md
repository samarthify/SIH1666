# CareerGuide

CareerGuide is an intelligent, user-friendly web platform designed to help students discover their ideal career paths. By leveraging advanced machine learning models and a rich dataset, CareerGuide provides personalized career suggestions based on your interests, strengths, and academic performance.

## Features
- Modern, responsive UI with Bootstrap 5 and custom design
- Personalized career suggestions using a trained XGBoost ML model
- Easy-to-use form for inputting interests, strengths, and academic details
- Results page with celebratory confetti and clear feedback
- About page describing the project and its mission
- SQLite database for storing user submissions

## Demo
Run the app locally and visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd SIH1666
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the ML Model
- Ensure `student_career_guidance_randomized.csv` is present in the project root.
- Run the ML training script to generate the model and preprocessors:
```bash
python xgb.py
```
This will create `xgb_model.pkl`, `encoder.pkl`, `scaler.pkl`, and `label_encoder.pkl`.

### 4. Run the App
```bash
python app.py
```

### 5. Access the App
Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Requirements
- Python 3.8+
- Flask
- Flask-SQLAlchemy
- pandas
- numpy
- scikit-learn
- xgboost
- joblib

Install all requirements with:
```bash
pip install -r requirements.txt
```

## Project Structure
```
SIH1666/
├── app.py
├── database.py
├── models.py
├── xgb.py
├── student_career_guidance_randomized.csv
├── students.db
├── requirements.txt
├── README.md
├── templates/
│   ├── home.html
│   ├── results.html
│   └── about.html
└── static/
    └── (optional JS/CSS files)
```

## Credits
Created by a passionate team for Smart India Hackathon 2025.

## License
This project is for educational and demonstration purposes. 