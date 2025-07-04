from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define the StudentData model
class StudentData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    interests = db.Column(db.String(200), nullable=False)
    dynamic_interests = db.Column(db.String(200), nullable=True)
    strengths = db.Column(db.String(200), nullable=False)
    academic_performance = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    language = db.Column(db.String(50), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

# Load the trained model and preprocessors
model = joblib.load('xgb_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Get form data
        interests = ','.join(request.form.getlist('interests'))
        dynamic_interests = request.form['dynamic_interests']
        strengths = request.form['strengths']
        academic_performance = request.form['academic_performance']
        age = request.form['age']
        location = request.form['location']
        language = request.form['language']

        # Save to database
        student_data = StudentData(
            interests=interests,
            dynamic_interests=dynamic_interests,
            strengths=strengths,
            academic_performance=academic_performance,
            age=age,
            location=location,
            language=language
        )
        db.session.add(student_data)
        db.session.commit()

        # Prepare input for the model
        input_interest = interests.split(',')[0] if interests else ''
        input_strength = strengths
        try:
            input_academic = float(academic_performance)
        except ValueError:
            return render_template('home.html', error='Please enter a valid number for Academic Performance.')
        # One-hot encode interests and strengths
        try:
            encoded_input = encoder.transform([[input_interest, input_strength]])
        except Exception as e:
            return render_template('home.html', error=f'Invalid input for strengths or interests: {e}')
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['Interest', 'Strength']))
        # Standardize academic performance
        academic_perf_scaled = scaler.transform([[input_academic]])
        encoded_input_df['Academic Performance'] = academic_perf_scaled[0]
        # Predict
        prediction = model.predict(encoded_input_df)
        career = label_encoder.inverse_transform(prediction)[0]
        # Redirect to the results page with the suggestion
        return redirect(url_for('results', suggestion=career))

@app.route('/results')
def results():
    suggestion = request.args.get('suggestion')
    return render_template('results.html', suggestion=suggestion)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
